//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import Foundation
import Observation
import QwenLocalCore

/// Training LoRA di modelli immagine (SD/SDXL) in stile DreamBooth:
/// una cartella di immagini + una frase con parola-trigger → un LoRA
/// utilizzabile in `/img` (e compatibile con ComfyUI).
///
/// Usa lo script ufficiale di diffusers (pinnato alla versione installata)
/// sul runtime integrato, GPU Metal via MPS. Onestà sui tempi: un LoRA
/// serio (800+ passi a 768/1024px) richiede ORE su Mac.
@MainActor
@Observable
final class ImageTrainingManager {

    enum State: Equatable {
        case idle
        case preparingTools
        case preparingBase
        case training(step: Int, total: Int)
        case done(loraPath: String)
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var log = ""

    // Input
    var baseCheckpoint: URL?
    var datasetDir: URL?
    var triggerPrompt = "a photo of zxq"
    var steps: Double = 800
    var rank: Double = 8
    var resolution = 768

    private(set) var datasetImageCount = 0
    private(set) var lastLoraURL: URL?

    private var runningProcess: Process?

    var isBusy: Bool {
        switch state {
        case .preparingTools, .preparingBase, .training: true
        default: false
        }
    }

    // MARK: - Percorsi (condivisi con lo stack immagini)

    private static var supportDir: URL { ImageGenManager.supportDir }
    private static var packagesDir: URL { ImageGenManager.packagesDir }
    private static var scriptURL: URL { supportDir.appendingPathComponent("train_lora_sdxl.py") }
    private static var baseModelsDir: URL {
        supportDir.appendingPathComponent("base-models", isDirectory: true)
    }
    static var lorasDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAiTraining/loras", isDirectory: true)
    }

    /// URL dello script di training pinnato alla versione di diffusers installata.
    private static let scriptRemote =
        "https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/examples/dreambooth/train_dreambooth_lora_sdxl.py"

    func setDataset(_ url: URL) {
        datasetDir = url
        let images = (try? FileManager.default.contentsOfDirectory(atPath: url.path))?
            .filter { ["png", "jpg", "jpeg", "webp"].contains(($0 as NSString).pathExtension.lowercased()) }
        datasetImageCount = images?.count ?? 0
    }

    // MARK: - Training

    func startTraining() {
        guard !isBusy else { return }
        guard let checkpoint = baseCheckpoint else {
            appendLog("✗ scegli il checkpoint base (.safetensors SDXL)\n")
            return
        }
        guard let dataset = datasetDir, datasetImageCount > 0 else {
            appendLog("✗ scegli una cartella con almeno qualche immagine\n")
            return
        }
        let trigger = triggerPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trigger.isEmpty else {
            appendLog("✗ serve la frase con la parola-trigger (es. \"a photo of zxq\")\n")
            return
        }
        guard let python = TrainingManager.embeddedPython else {
            state = .failed("runtime Python integrato assente")
            return
        }
        // Solo SDXL in v1: lo script pinnato è quello SDXL.
        let report = SafetensorsInspector.inspect(url: checkpoint)
        guard report.contains("Stable Diffusion XL") else {
            state = .failed("il training LoRA integrato supporta checkpoint SDXL (questo non lo è)")
            return
        }

        Task {
            // 1. strumenti (peft + script) — no-op se già presenti
            state = .preparingTools
            if !FileManager.default.fileExists(atPath: Self.scriptURL.path) {
                appendLog("✻ scarico lo script di training (diffusers 0.39.0)…\n")
                guard let (data, _) = try? await URLSession.shared.data(
                    from: URL(string: Self.scriptRemote)!),
                    (try? data.write(to: Self.scriptURL)) != nil
                else {
                    state = .failed("download dello script fallito")
                    return
                }
            }
            let peftCheck = await runStreaming(python.path, ["-c", "import peft, torchvision"], quiet: true)
            if peftCheck != 0 {
                appendLog("✻ installo peft (una tantum)…\n")
                let install = await runStreaming(python.path, [
                    "-m", "pip", "install", "--quiet",
                    "--target", Self.packagesDir.path, "peft", "torchvision",
                ])
                guard install == 0 else {
                    state = .failed("installazione peft fallita")
                    return
                }
            }

            // 2. base in formato diffusers (cache per checkpoint)
            let baseName = checkpoint.deletingPathExtension().lastPathComponent
            let baseDir = Self.baseModelsDir.appendingPathComponent(baseName, isDirectory: true)
            if !FileManager.default.fileExists(
                atPath: baseDir.appendingPathComponent("model_index.json").path) {
                state = .preparingBase
                appendLog("✻ converto \(checkpoint.lastPathComponent) in formato diffusers (una tantum, ~7 GB)…\n")
                let convert = await runStreaming(python.path, [
                    "-c",
                    """
                    import torch
                    from diffusers import StableDiffusionXLPipeline
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        "\(checkpoint.path)", torch_dtype=torch.float16, use_safetensors=True)
                    pipe.save_pretrained("\(baseDir.path)")
                    """,
                ])
                guard convert == 0 else {
                    state = .failed("conversione della base fallita — vedi log")
                    return
                }
            }

            // 3. training
            let stamp = Int(Date.timeIntervalSinceReferenceDate)
            let outputDir = Self.lorasDir.appendingPathComponent("lora-\(baseName)-\(stamp)")
            try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
            state = .training(step: 0, total: Int(steps))
            appendLog("""
                ✻ training LoRA — \(datasetImageCount) immagini · trigger "\(trigger)"
                  rank \(Int(rank)) · \(Int(steps)) passi · \(resolution)px
                  (~1.6 passi/secondo a 512px su questo Mac)\n
                """)
            let status = await runStreaming(python.path, [
                Self.scriptURL.path,
                "--pretrained_model_name_or_path", baseDir.path,
                "--instance_data_dir", dataset.path,
                "--instance_prompt", trigger,
                "--output_dir", outputDir.path,
                "--rank", String(Int(rank)),
                "--resolution", String(resolution),
                "--train_batch_size", "1",
                "--gradient_accumulation_steps", "1",
                "--max_train_steps", String(Int(steps)),
                "--learning_rate", "1e-4",
                "--lr_scheduler", "constant",
                "--lr_warmup_steps", "0",
                "--mixed_precision", "no",
                "--seed", "42",
            ])
            let lora = outputDir.appendingPathComponent("pytorch_lora_weights.safetensors")
            if status == 0, FileManager.default.fileExists(atPath: lora.path) {
                lastLoraURL = lora
                state = .done(loraPath: lora.path)
                appendLog("\n✓ LoRA pronto: \(lora.path)\n  usalo in /img (pannello parametri → LoRA) o in ComfyUI\n")
            } else {
                state = .failed("training terminato con exit \(status) — vedi log")
            }
        }
    }

    func cancel() {
        runningProcess?.terminate()
        runningProcess = nil
        if isBusy { state = .idle }
        appendLog("\n⏹ training interrotto\n")
    }

    // MARK: - Runner con progresso

    private func runStreaming(_ executable: String, _ arguments: [String], quiet: Bool = false) async -> Int32 {
        await withCheckedContinuation { continuation in
            nonisolated(unsafe) let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = Self.packagesDir.path
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            environment["TERM"] = "dumb"
            process.environment = environment

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            process.standardInput = FileHandle.nullDevice
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if !quiet {
                        self.appendLog(text)
                    }
                    self.parseProgress(text)
                }
            }
            process.terminationHandler = { finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                let status = finished.terminationStatus
                Task { @MainActor [weak self] in
                    self?.runningProcess = nil
                    continuation.resume(returning: status)
                }
            }
            do {
                try process.run()
                runningProcess = process
            } catch {
                appendLog("✗ avvio fallito: \(error.localizedDescription)\n")
                continuation.resume(returning: -1)
            }
        }
    }

    /// Estrae il passo corrente dalle righe tqdm dello script ("Steps: … 12/800 …").
    private func parseProgress(_ text: String) {
        guard case .training(_, let total) = state else { return }
        for line in text.split(separator: "\r").flatMap({ $0.split(separator: "\n") }) {
            guard line.contains("Steps"),
                let match = line.firstMatch(of: /(\d+)\/(\d+)/),
                let step = Int(match.1)
            else { continue }
            state = .training(step: step, total: Int(match.2) ?? total)
        }
    }

    private func appendLog(_ text: String) {
        // le barre tqdm usano \r: tieni solo l'ultimo aggiornamento della riga
        let cleaned = text.split(separator: "\r").last.map(String.init) ?? text
        log += cleaned.hasSuffix("\n") ? cleaned : cleaned + "\n"
        if log.count > 150_000 {
            log = String(log.suffix(100_000))
        }
    }
}
