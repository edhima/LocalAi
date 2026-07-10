//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import Foundation
import Observation
import PDFKit

/// Gestisce il fine-tuning LoRA/QLoRA dei modelli Qwen con lo stack ufficiale
/// MLX (`mlx-lm`, Python), installato in un venv dedicato dell'app.
///
/// Pipeline: import file grezzi → preparazione dataset via modello locale
/// (JSONL formato chat, split train/valid) → `mlx_lm.lora` → `mlx_lm.fuse`
/// → caricamento del modello fuso in chat.
@MainActor
@Observable
final class TrainingManager {

    // MARK: - Stato

    private(set) var toolsReady = false
    private(set) var isBusy = false
    private(set) var busyLabel = ""
    /// Modalità mostrata dalla finestra Training: 0 = testo (LLM), 1 = immagini (LoRA SDXL).
    var windowMode = 0

    var rawFiles: [URL] = []
    private(set) var trainCount = 0
    private(set) var validCount = 0
    private(set) var preparedChunks = 0
    private(set) var totalChunks = 0

    // Parametri di training (default ragionevoli per QLoRA su 4B)
    var trainModelID = "mlx-community/Qwen3-4B-4bit"
    var iters: Double = 600
    var batchSize: Double = 2
    var learningRate: Double = 1e-5
    var examplesPerChunk = 3

    private(set) var currentIter = 0
    private(set) var lastTrainLoss: Double?
    private(set) var lastValLoss: Double?
    private(set) var log = ""

    private(set) var hasAdapters = false
    private(set) var fusedModelURL: URL?
    private(set) var fp16ExportURL: URL?
    private(set) var ggufExportURL: URL?

    private var runningProcess: Process?
    private var prepTask: Task<Void, Never>?

    // MARK: - Percorsi

    /// Cartella del progetto di training corrente.
    private(set) var projectURL: URL

    static var baseDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAiTraining", isDirectory: true)
    }

    static var venvDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAi/venv", isDirectory: true)
    }

    var dataDir: URL { projectURL.appendingPathComponent("data", isDirectory: true) }
    var adaptersDir: URL { projectURL.appendingPathComponent("adapters", isDirectory: true) }
    var fusedDir: URL { projectURL.appendingPathComponent("fused", isDirectory: true) }
    var exportFP16Dir: URL { projectURL.appendingPathComponent("export-fp16", isDirectory: true) }
    var exportGGUFDir: URL { projectURL.appendingPathComponent("export-gguf", isDirectory: true) }

    private var venvBin: URL { Self.venvDir.appendingPathComponent("bin") }

    /// Python integrato nel bundle (Contents/Resources/python) con lo stack ML:
    /// l'app è autonoma e la disinstallazione è "cestina LocalAi.app".
    /// `python3` è un symlink a `python3.12`: proviamo entrambi, e verifichiamo
    /// che il file esista davvero (più robusto di `isExecutableFile` sui symlink).
    static var embeddedPython: URL? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        let fm = FileManager.default
        for name in ["python/bin/python3", "python/bin/python3.12"] {
            let candidate = resources.appendingPathComponent(name)
            let resolved = candidate.resolvingSymlinksInPath()
            if fm.fileExists(atPath: resolved.path) || fm.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }
    var usesEmbeddedRuntime: Bool { Self.embeddedPython != nil }

    /// Comando per un modulo mlx_lm: runtime integrato se presente, altrimenti venv legacy.
    private func mlxCommand(_ module: String, _ args: [String]) -> (String, [String]) {
        if let python = Self.embeddedPython {
            return (python.path, ["-m", module] + args)
        }
        return (venvBin.appendingPathComponent(module).path, args)
    }

    init() {
        let stamp = ISO8601DateFormatter().string(from: .now)
            .replacingOccurrences(of: ":", with: "-")
        projectURL = Self.baseDir.appendingPathComponent("progetto-\(stamp)", isDirectory: true)
        try? FileManager.default.createDirectory(at: projectURL, withIntermediateDirectories: true)
        refreshEnvironment()
    }

    func refreshEnvironment() {
        toolsReady = usesEmbeddedRuntime
            || FileManager.default.isExecutableFile(
                atPath: venvBin.appendingPathComponent("mlx_lm.lora").path)
        hasAdapters = FileManager.default.fileExists(
            atPath: adaptersDir.appendingPathComponent("adapters.safetensors").path)
        let fusedConfig = fusedDir.appendingPathComponent("config.json")
        fusedModelURL = FileManager.default.fileExists(atPath: fusedConfig.path) ? fusedDir : nil
        let fp16Config = exportFP16Dir.appendingPathComponent("config.json")
        fp16ExportURL = FileManager.default.fileExists(atPath: fp16Config.path) ? exportFP16Dir : nil
        ggufExportURL = (try? FileManager.default.contentsOfDirectory(
            at: exportGGUFDir, includingPropertiesForKeys: nil))?
            .first { $0.pathExtension == "gguf" }
        refreshDatasetCounts()
    }

    private func refreshDatasetCounts() {
        trainCount = Self.lineCount(dataDir.appendingPathComponent("train.jsonl"))
        validCount = Self.lineCount(dataDir.appendingPathComponent("valid.jsonl"))
    }

    private static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(separator: "\n", omittingEmptySubsequences: true).count
    }

    // MARK: - Setup strumenti (venv + mlx-lm)

    func setupTools() {
        guard !isBusy else { return }
        isBusy = true
        busyLabel = "installo mlx-lm…"
        appendLog("✻ creo il venv Python e installo mlx-lm (una tantum)…\n")
        Task {
            let venv = Self.venvDir
            try? FileManager.default.createDirectory(
                at: venv.deletingLastPathComponent(), withIntermediateDirectories: true)
            var status = await run("/usr/bin/python3", ["-m", "venv", venv.path])
            if status == 0 {
                status = await run(
                    venvBin.appendingPathComponent("pip").path,
                    ["install", "--upgrade", "mlx-lm"])
            }
            if status == 0 {
                appendLog("\n✓ strumenti pronti (venv di fallback)\n")
            } else {
                appendLog("\n✗ installazione fallita (exit \(status)). Serve Python 3 (Xcode CLT).\n")
            }
            refreshEnvironment()
            isBusy = false
        }
    }

    // MARK: - Import e preparazione dati

    func addRawFiles(_ urls: [URL]) {
        for url in urls where !rawFiles.contains(url) {
            // I JSONL già in formato chat vanno diretti nel dataset.
            if url.pathExtension.lowercased() == "jsonl" {
                importReadyJSONL(url)
            } else {
                rawFiles.append(url)
            }
        }
    }

    /// Un JSONL già pronto (righe {"messages": […]}) viene validato e splittato.
    private func importReadyJSONL(_ url: URL) {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        let valid = text.split(separator: "\n").map(String.init).filter(Self.isValidExample)
        appendLog("✻ \(url.lastPathComponent): \(valid.count) esempi validi importati direttamente\n")
        writeDataset(examples: valid, append: true)
    }

    /// Estrae il testo da un file grezzo (txt, md, csv, json, pdf…).
    static func extractText(_ url: URL) -> String? {
        if url.pathExtension.lowercased() == "pdf" {
            return PDFDocument(url: url)?.string
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// Il modello locale trasforma i file grezzi in esempi di training.
    func prepareData(engine: QwenEngine) {
        guard !isBusy, !rawFiles.isEmpty else { return }
        guard engine.phase == .ready else {
            appendLog("✗ carica prima un modello in chat: serve per generare gli esempi\n")
            return
        }
        isBusy = true
        busyLabel = "preparo il dataset…"

        // Spezza tutti i file in blocchi da ~1800 caratteri.
        var chunks: [String] = []
        for file in rawFiles {
            guard let text = Self.extractText(file) else {
                appendLog("✗ non riesco a leggere \(file.lastPathComponent)\n")
                continue
            }
            chunks.append(contentsOf: Self.split(text: text, size: 1800))
        }
        totalChunks = chunks.count
        preparedChunks = 0
        appendLog("✻ preparazione: \(chunks.count) blocchi da \(rawFiles.count) file, \(examplesPerChunk) esempi/blocco\n")

        let perChunk = examplesPerChunk
        prepTask = Task {
            var examples: [String] = []
            for chunk in chunks {
                if Task.isCancelled { break }
                // Sessione fresca per ogni blocco: niente accumulo di contesto.
                guard let session = engine.utilitySession(
                    instructions: "You generate supervised fine-tuning data. Output ONLY JSONL lines, nothing else.")
                else { break }
                let prompt = """
                    From the following text, write exactly \(perChunk) training examples as JSONL. \
                    One JSON object per line, each: {"messages": [{"role": "user", "content": "..."}, \
                    {"role": "assistant", "content": "..."}]}. Questions must be answerable from the \
                    text, varied, in the same language as the text. Answers complete but concise. \
                    Output only the \(perChunk) JSONL lines.

                    TEXT:
                    \(chunk)
                    """
                do {
                    var output = ""
                    for try await chunk in session.streamResponse(to: prompt) {
                        output += chunk
                    }
                    let lines = output.split(separator: "\n").map(String.init)
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter(Self.isValidExample)
                    examples.append(contentsOf: lines)
                    preparedChunks += 1
                    if preparedChunks % 5 == 0 || preparedChunks == totalChunks {
                        appendLog("  blocco \(preparedChunks)/\(totalChunks) — \(examples.count) esempi validi\n")
                    }
                } catch {
                    appendLog("  ✗ blocco \(preparedChunks + 1): \(error.localizedDescription)\n")
                    preparedChunks += 1
                }
            }
            writeDataset(examples: examples, append: true)
            appendLog("✓ dataset: \(trainCount) train / \(validCount) valid in \(dataDir.path)\n")
            isBusy = false
        }
    }

    /// Una riga è un esempio valido se è JSON con "messages" = array di ≥ 2 turni.
    static func isValidExample(_ line: String) -> Bool {
        guard line.hasPrefix("{"), let data = line.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let messages = object["messages"] as? [[String: Any]], messages.count >= 2
        else { return false }
        return messages.allSatisfy { $0["role"] is String && $0["content"] is String }
    }

    /// Scrive train.jsonl / valid.jsonl con split 90/10 (almeno 1 riga in valid).
    private func writeDataset(examples: [String], append: Bool) {
        guard !examples.isEmpty else { return }
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        var train = (append ? Self.readLines(dataDir.appendingPathComponent("train.jsonl")) : [])
        var valid = (append ? Self.readLines(dataDir.appendingPathComponent("valid.jsonl")) : [])
        for (index, example) in examples.enumerated() {
            if (index + train.count + valid.count) % 10 == 9 {
                valid.append(example)
            } else {
                train.append(example)
            }
        }
        if valid.isEmpty, train.count > 1 {
            valid.append(train.removeLast())
        }
        try? train.joined(separator: "\n").appending("\n")
            .write(to: dataDir.appendingPathComponent("train.jsonl"), atomically: true, encoding: .utf8)
        try? valid.joined(separator: "\n").appending("\n")
            .write(to: dataDir.appendingPathComponent("valid.jsonl"), atomically: true, encoding: .utf8)
        refreshDatasetCounts()
    }

    private static func readLines(_ url: URL) -> [String] {
        (try? String(contentsOf: url, encoding: .utf8))?
            .split(separator: "\n", omittingEmptySubsequences: true).map(String.init) ?? []
    }

    static func split(text: String, size: Int) -> [String] {
        let paragraphs = text.components(separatedBy: "\n\n")
        var chunks: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if current.count + paragraph.count > size, !current.isEmpty {
                chunks.append(current)
                current = ""
            }
            current += (current.isEmpty ? "" : "\n\n") + paragraph
        }
        if current.trimmingCharacters(in: .whitespacesAndNewlines).count > 80 {
            chunks.append(current)
        }
        return chunks
    }

    // MARK: - Conversione HF → MLX

    /// Converte un modello Hugging Face (id del hub o cartella locale in formato HF)
    /// in un modello MLX pronto per il training LoRA e per la chat.
    /// `quantizeBits`: 4 o 8 per quantizzare, nil per tenere la precisione originale.
    func convertModel(source: String, quantizeBits: Int?) {
        guard !isBusy, toolsReady else { return }
        let clean = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        // Un singolo .safetensors non è convertibile: servono config e tokenizer.
        if clean.hasSuffix(".safetensors") {
            appendLog("✗ un singolo .safetensors non basta: servono config.json e tokenizer. Indica la cartella completa del modello (o l'id Hugging Face).\n")
            return
        }

        let name = clean.split(separator: "/").last.map(String.init) ?? clean
        let suffix = quantizeBits.map { "-\($0)bit" } ?? "-mlx"
        let destination = Self.baseDir
            .appendingPathComponent("converted", isDirectory: true)
            .appendingPathComponent(name + suffix, isDirectory: true)

        isBusy = true
        busyLabel = "converto in MLX…"
        appendLog("✻ mlx_lm.convert \(clean) → \(destination.path)\(quantizeBits.map { " (\($0)-bit)" } ?? "")\n")
        Task {
            try? FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)

            // Per gli id del hub: snapshot COMPLETO prima della conversione
            // (mlx_lm scarica solo i pesi e huggingface_hub poi rifiuta lo
            // snapshot parziale al salvataggio).
            if !clean.hasPrefix("/"), clean.contains("/"), let python = TrainingManager.embeddedPython {
                appendLog("✻ scarico lo snapshot completo di \(clean)…\n")
                let dl = await run(python.path, [
                    "-c",
                    "from huggingface_hub import snapshot_download; snapshot_download('\(clean)')",
                ])
                guard dl == 0 else {
                    appendLog("\n✗ download fallito (exit \(dl))\n")
                    isBusy = false
                    return
                }
            }

            var args = ["--hf-path", clean, "--mlx-path", destination.path]
            if let bits = quantizeBits {
                args += ["-q", "--q-bits", String(bits)]
            }
            let (tool, fullArgs) = mlxCommand("mlx_lm.convert", args)
            let status = await run(tool, fullArgs)
            if status == 0,
                FileManager.default.fileExists(
                    atPath: destination.appendingPathComponent("config.json").path) {
                trainModelID = destination.path
                appendLog("""
                    \n✓ modello MLX pronto: \(destination.path)
                      · impostato come modello di training (passo 3)
                      · montabile in chat: sidebar → "cartella modello locale…"\n
                    """)
            } else {
                appendLog("\n✗ conversione fallita (exit \(status)) — vedi log sopra\n")
            }
            refreshEnvironment()
            isBusy = false
        }
    }

    // MARK: - Training e fusione

    func startTraining() {
        guard !isBusy, toolsReady, trainCount > 0 else { return }
        isBusy = true
        busyLabel = "training in corso…"
        currentIter = 0
        lastTrainLoss = nil
        lastValLoss = nil
        appendLog("""
            ✻ mlx_lm.lora — modello \(trainModelID)
              iters \(Int(iters)) · batch \(Int(batchSize)) · lr \(learningRate)
            """ + "\n")
        Task {
            let (tool, args) = mlxCommand("mlx_lm.lora", [
                "--model", trainModelID,
                "--train",
                "--data", dataDir.path,
                "--iters", String(Int(iters)),
                "--batch-size", String(Int(batchSize)),
                "--learning-rate", String(learningRate),
                "--adapter-path", adaptersDir.path,
                "--save-every", "100",
            ])
            let status = await run(tool, args)
            appendLog(status == 0
                ? "\n✓ training completato — adapters in \(adaptersDir.path)\n"
                : "\n✗ training terminato con exit \(status)\n")
            refreshEnvironment()
            isBusy = false
        }
    }

    func fuse() {
        guard !isBusy, toolsReady, hasAdapters else { return }
        isBusy = true
        busyLabel = "fondo gli adapter…"
        appendLog("✻ mlx_lm.fuse → \(fusedDir.path)\n")
        Task {
            let (tool, args) = mlxCommand("mlx_lm.fuse", [
                "--model", trainModelID,
                "--adapter-path", adaptersDir.path,
                "--save-path", fusedDir.path,
            ])
            let status = await run(tool, args)
            appendLog(status == 0
                ? "\n✓ modello fuso pronto: \(fusedDir.path)\n"
                : "\n✗ fusione fallita (exit \(status))\n")
            refreshEnvironment()
            isBusy = false
        }
    }

    /// Esporta il modello fuso dequantizzato in float16 (safetensors HF standard):
    /// caricabile direttamente da vLLM, TGI e transformers su sistemi live.
    func exportFP16() {
        guard !isBusy, toolsReady, hasAdapters else { return }
        isBusy = true
        busyLabel = "esporto fp16 per vLLM…"
        appendLog("✻ mlx_lm.fuse --dequantize → \(exportFP16Dir.path)\n")
        Task {
            let (tool, args) = mlxCommand("mlx_lm.fuse", [
                "--model", trainModelID,
                "--adapter-path", adaptersDir.path,
                "--save-path", exportFP16Dir.path,
                "--dequantize",
            ])
            let status = await run(tool, args)
            appendLog(status == 0
                ? "\n✓ export fp16 pronto (vLLM/TGI/transformers): \(exportFP16Dir.path)\n"
                : "\n✗ export fp16 fallito (exit \(status))\n")
            refreshEnvironment()
            isBusy = false
        }
    }

    /// Esporta in GGUF (Ollama / llama.cpp). Il supporto dipende dall'architettura:
    /// se mlx-lm non la supporta, la via è export fp16 + convert_hf_to_gguf di llama.cpp.
    func exportGGUF() {
        guard !isBusy, toolsReady, hasAdapters else { return }
        isBusy = true
        busyLabel = "esporto GGUF…"
        try? FileManager.default.createDirectory(at: exportGGUFDir, withIntermediateDirectories: true)
        let ggufFile = exportGGUFDir.appendingPathComponent("model-f16.gguf")
        appendLog("✻ mlx_lm.fuse --export-gguf → \(ggufFile.path)\n")
        Task {
            let (tool, args) = mlxCommand("mlx_lm.fuse", [
                "--model", trainModelID,
                "--adapter-path", adaptersDir.path,
                "--save-path", fusedDir.path,
                "--export-gguf",
                "--gguf-path", ggufFile.path,
            ])
            let status = await run(tool, args)
            if status == 0, FileManager.default.fileExists(atPath: ggufFile.path) {
                appendLog("\n✓ GGUF pronto (Ollama/llama.cpp): \(ggufFile.path)\n")
            } else {
                appendLog("""
                    \n✗ export GGUF fallito: probabilmente l'architettura non è supportata \
                    dall'export diretto di mlx-lm (es. Qwen3). Alternativa: esporta fp16 e \
                    converti con convert_hf_to_gguf.py di llama.cpp.\n
                    """)
            }
            refreshEnvironment()
            isBusy = false
        }
    }

    func cancel() {
        prepTask?.cancel()
        prepTask = nil
        runningProcess?.terminate()
        appendLog("\n⏹ interrotto dall'utente\n")
        isBusy = false
    }

    // MARK: - Esecuzione processi con log in streaming

    private func run(_ executable: String, _ arguments: [String]) async -> Int32 {
        await withCheckedContinuation { continuation in
            nonisolated(unsafe) let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            var environment = ProcessInfo.processInfo.environment
            environment["PATH"] = "\(venvBin.path):/usr/bin:/bin:/usr/sbin:/sbin"
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
                    self?.appendLog(text)
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
                appendLog("✗ impossibile avviare \(executable): \(error.localizedDescription)\n")
                continuation.resume(returning: -1)
            }
        }
    }

    private func appendLog(_ text: String) {
        log += text
        if log.count > 200_000 {
            log = String(log.suffix(150_000))
        }
        parseProgress(text)
    }

    /// Estrae iterazione e loss dalle righe di mlx_lm.lora
    /// (es. "Iter 100: Train loss 1.234, …" / "Iter 100: Val loss 1.456, …").
    private func parseProgress(_ text: String) {
        for line in text.split(separator: "\n") {
            guard let iterMatch = line.firstMatch(of: /Iter (\d+):/) ,
                let iter = Int(iterMatch.1) else { continue }
            currentIter = max(currentIter, iter)
            if let match = line.firstMatch(of: /Train loss ([\d.]+)/) {
                lastTrainLoss = Double(match.1)
            }
            if let match = line.firstMatch(of: /Val loss ([\d.]+)/) {
                lastValLoss = Double(match.1)
            }
        }
    }
}