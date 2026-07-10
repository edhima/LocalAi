//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import QwenLocalCore
import Foundation
import Observation

/// Generazione immagini da checkpoint Stable Diffusion / SDXL single-file
/// (es. un .safetensors da ComfyUI), via diffusers su GPU Metal (MPS).
///
/// Architettura: un worker Python persistente carica la pipeline una volta
/// sola e riceve richieste JSON su stdin; le risposte arrivano su stdout.
/// Lo stack (torch + diffusers, ~2,5 GB) vive in Application Support/LocalAi/imagegen
/// e si installa on-demand.
@MainActor
@Observable
final class ImageGenManager {

    enum State: Equatable {
        case toolsMissing
        case installingTools
        case idle                    // tools ok, nessun modello montato
        case loadingModel
        case ready(model: String)
        case generating(step: Int, total: Int)
        case failed(String)
    }

    private(set) var state: State = .toolsMissing
    private(set) var modelURL: URL?
    private(set) var log = ""

    private var worker: Process?
    private var workerStdin: FileHandle?
    private var pendingCompletion: ((Result<URL, Error>) -> Void)?
    private var imageCounter = 0

    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }
    var isBusy: Bool {
        switch state {
        case .installingTools, .loadingModel, .generating: true
        default: false
        }
    }

    // MARK: - Percorsi

    static var supportDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAi/imagegen", isDirectory: true)
    }
    /// I pacchetti Python dello stack immagini (pip --target): nessun venv,
    /// il worker usa il Python integrato nell'app + PYTHONPATH.
    static var packagesDir: URL { supportDir.appendingPathComponent("packages", isDirectory: true) }
    static var outputDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAiImages", isDirectory: true)
    }
    private static var workerScript: URL { supportDir.appendingPathComponent("worker.py") }

    init() {
        refreshEnvironment()
    }

    func refreshEnvironment() {
        if TrainingManager.embeddedPython != nil, markerOK {
            if case .toolsMissing = state { state = .idle }
        } else if !isBusy {
            state = .toolsMissing
        }
    }

    private var markerOK: Bool {
        FileManager.default.fileExists(
            atPath: Self.supportDir.appendingPathComponent(".ready").path)
    }

    // MARK: - Installazione stack immagini (torch + diffusers)

    func installTools() {
        guard !isBusy else { return }
        state = .installingTools
        appendLog("✻ installo lo stack immagini (torch + diffusers, ~2,5 GB — una tantum)…\n")
        Task {
            guard let embedded = TrainingManager.embeddedPython else {
                state = .failed("runtime Python integrato assente")
                return
            }
            try? FileManager.default.createDirectory(
                at: Self.packagesDir, withIntermediateDirectories: true)
            // pip --target: solo librerie, nessun binario copiato → sopravvive
            // a spostamenti/aggiornamenti dell'app.
            let status = await runOnce(embedded.path, [
                "-m", "pip", "install", "--quiet",
                "--target", Self.packagesDir.path,
                "torch", "diffusers", "transformers~=5.12.0",
                "safetensors", "accelerate", "pillow",
            ])
            if status == 0 {
                Self.writeWorkerScript()
                FileManager.default.createFile(
                    atPath: Self.supportDir.appendingPathComponent(".ready").path, contents: nil)
                appendLog("✓ stack immagini pronto\n")
                state = .idle
            } else {
                state = .failed("installazione fallita (exit \(status)) — vedi log")
            }
        }
    }

    /// Rimuove lo stack immagini dal disco (recupera ~2,5 GB).
    func uninstallTools() {
        unmount()
        try? FileManager.default.removeItem(at: Self.supportDir)
        state = .toolsMissing
        appendLog("✓ stack immagini rimosso\n")
    }

    // MARK: - Montaggio modello e worker

    /// Monta un checkpoint SD/SDXL single-file e avvia il worker (carica la pipeline).
    func mount(checkpoint: URL) {
        guard !isBusy else { return }
        if case .toolsMissing = state {
            appendLog("✗ installa prima lo stack immagini\n")
            return
        }
        actuallyMount(checkpoint)
    }

    private func actuallyMount(_ checkpoint: URL) {
        unmount()
        // valida col referto dell'ispettore: accettiamo solo checkpoint SD/SDXL
        let report = SafetensorsInspector.inspect(url: checkpoint)
        guard report.contains("Stable Diffusion") else {
            state = .failed("non è un checkpoint Stable Diffusion/SDXL: \(checkpoint.lastPathComponent)")
            return
        }
        let isXL = report.contains("XL")
        Self.writeWorkerScript()
        modelURL = checkpoint
        state = .loadingModel
        appendLog("✻ carico \(checkpoint.lastPathComponent) (\(isXL ? "SDXL" : "SD"))…\n")

        guard let embedded = TrainingManager.embeddedPython else {
            state = .failed("runtime Python integrato assente")
            return
        }
        let process = Process()
        process.executableURL = embedded
        process.arguments = [Self.workerScript.path, checkpoint.path, isXL ? "xl" : "sd"]
        process.currentDirectoryURL = Self.supportDir
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = Self.packagesDir.path
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.handleWorkerOutput(text)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.appendLog(text)
            }
        }
        process.terminationHandler = { finished in
            let status = finished.terminationStatus
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.worker = nil
                self.workerStdin = nil
                if case .loadingModel = self.state {
                    self.state = .failed("il worker è terminato durante il caricamento (exit \(status)) — vedi log")
                } else if case .generating = self.state {
                    self.pendingCompletion?(.failure(AgentToolbox.ToolboxError("worker terminato")))
                    self.pendingCompletion = nil
                    self.state = .failed("worker terminato (exit \(status))")
                }
            }
        }

        do {
            try process.run()
            worker = process
            workerStdin = stdinPipe.fileHandleForWriting
        } catch {
            state = .failed("impossibile avviare il worker: \(error.localizedDescription)")
        }
    }

    func unmount() {
        worker?.terminate()
        worker = nil
        workerStdin = nil
        modelURL = nil
        pendingCompletion = nil
        if markerOK { state = .idle }
    }

    // MARK: - Generazione

    func generate(prompt: String, completion: @escaping (Result<URL, Error>) -> Void) {
        guard isReady, let workerStdin else {
            completion(.failure(AgentToolbox.ToolboxError("nessun modello immagine montato")))
            return
        }
        guard pendingCompletion == nil else {
            completion(.failure(AgentToolbox.ToolboxError("generazione già in corso")))
            return
        }
        try? FileManager.default.createDirectory(
            at: Self.outputDir, withIntermediateDirectories: true)
        imageCounter += 1
        let stamp = Int(Date.timeIntervalSinceReferenceDate)
        let output = Self.outputDir.appendingPathComponent("img-\(stamp)-\(imageCounter).png")

        pendingCompletion = completion
        state = .generating(step: 0, total: 0)

        let request: [String: Any] = ["prompt": prompt, "out": output.path]
        if let data = try? JSONSerialization.data(withJSONObject: request) {
            workerStdin.write(data)
            workerStdin.write("\n".data(using: .utf8)!)
        }
    }

    // MARK: - Protocollo worker

    private var stdoutBuffer = ""

    private func handleWorkerOutput(_ text: String) {
        stdoutBuffer += text
        while let newline = stdoutBuffer.firstIndex(of: "\n") {
            let line = String(stdoutBuffer[..<newline])
            stdoutBuffer = String(stdoutBuffer[stdoutBuffer.index(after: newline)...])
            handleWorkerLine(line)
        }
    }

    private func handleWorkerLine(_ line: String) {
        guard let data = line.data(using: .utf8),
            let message = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let status = message["status"] as? String
        else {
            if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                appendLog(line + "\n")
            }
            return
        }
        switch status {
        case "ready":
            if let model = modelURL {
                state = .ready(model: model.lastPathComponent)
                appendLog("✓ pipeline pronta\n")
            }
        case "progress":
            let step = message["step"] as? Int ?? 0
            let total = message["total"] as? Int ?? 0
            state = .generating(step: step, total: total)
        case "done":
            if let path = message["out"] as? String, let model = modelURL {
                state = .ready(model: model.lastPathComponent)
                pendingCompletion?(.success(URL(fileURLWithPath: path)))
                pendingCompletion = nil
            }
        case "error":
            let reason = message["message"] as? String ?? "errore sconosciuto"
            appendLog("✗ \(reason)\n")
            if let model = modelURL {
                state = .ready(model: model.lastPathComponent)
            }
            pendingCompletion?(.failure(AgentToolbox.ToolboxError(reason)))
            pendingCompletion = nil
        default:
            break
        }
    }

    // MARK: - Utilità

    private func runOnce(_ executable: String, _ arguments: [String]) async -> Int32 {
        await withCheckedContinuation { continuation in
            nonisolated(unsafe) let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor [weak self] in self?.appendLog(text) }
            }
            process.terminationHandler = { finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                let status = finished.terminationStatus
                Task { @MainActor in continuation.resume(returning: status) }
            }
            do {
                try process.run()
            } catch {
                continuation.resume(returning: -1)
            }
        }
    }

    private func appendLog(_ text: String) {
        log += text
        if log.count > 100_000 {
            log = String(log.suffix(70_000))
        }
    }

    /// Il worker Python: carica la pipeline una volta, poi genera su richiesta.
    private static func writeWorkerScript() {
        let script = #"""
        import json, random, sys, torch
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionPipeline,
            EulerAncestralDiscreteScheduler,
        )

        checkpoint, arch = sys.argv[1], sys.argv[2]
        cls = StableDiffusionXLPipeline if arch == "xl" else StableDiffusionPipeline
        pipe = cls.from_single_file(checkpoint, torch_dtype=torch.float16, use_safetensors=True)
        # euler_ancestral: lo scheduler con cui questi checkpoint sono usati in ComfyUI
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
        print(json.dumps({"status": "ready"}), flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                steps = int(req.get("steps", 25))
                seed = int(req.get("seed", -1))
                if seed < 0:
                    seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator(device="mps").manual_seed(seed)

                def on_step(pipeline, step, timestep, kwargs):
                    print(json.dumps({"status": "progress", "step": step + 1, "total": steps}), flush=True)
                    return kwargs

                image = pipe(
                    req["prompt"],
                    negative_prompt=req.get("negative", ""),
                    num_inference_steps=steps,
                    guidance_scale=float(req.get("cfg", 7.0)),
                    width=int(req.get("width", 1024 if arch == "xl" else 512)),
                    height=int(req.get("height", 1024 if arch == "xl" else 512)),
                    generator=generator,
                    callback_on_step_end=on_step,
                ).images[0]
                image.save(req["out"])
                print(json.dumps({"status": "done", "out": req["out"], "seed": seed}), flush=True)
            except Exception as exc:
                print(json.dumps({"status": "error", "message": str(exc)}), flush=True)
        """#
        try? script.write(to: workerScript, atomically: true, encoding: .utf8)
    }
}