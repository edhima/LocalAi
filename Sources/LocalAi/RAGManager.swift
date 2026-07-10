import QwenLocalCore
import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Observation
import Tokenizers

/// Un frammento di documento indicizzato con il suo embedding.
struct RAGChunk: Codable {
    var file: String
    var text: String
    var vector: [Float]
}

/// Indice semantico del workspace: embedding dei documenti con
/// `multilingual-e5-small` (ottimo anche in italiano) e ricerca per coseno.
/// L'indice è persistito in `<workspace>/.localai/rag-index.json`.
@MainActor
@Observable
final class RAGManager {

    enum State: Equatable {
        case idle
        case loadingModel(fraction: Double)
        case indexing(done: Int, total: Int)
        case ready(chunks: Int)
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var indexedWorkspace: URL?

    private var container: EmbedderModelContainer?
    private var chunks: [RAGChunk] = []
    private var indexTask: Task<Void, Never>?

    /// Embedder multilingue, ~450 MB al primo download (stessa cache HF dei modelli chat).
    static let embedderConfiguration = EmbedderRegistry.multilingual_e5_small

    static let indexedExtensions: Set<String> = [
        "txt", "md", "markdown", "rst", "csv", "json", "yaml", "yml", "html", "tex",
        "swift", "py", "js", "ts", "java", "kt", "rb", "go", "rs", "c", "cpp", "h", "sh",
        "pdf",
    ]
    private static let excludedDirs: Set<String> = [
        ".git", ".build", "node_modules", ".localai", ".xcodebuild", ".tooling",
    ]

    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    // MARK: - Persistenza

    private static func indexURL(for workspace: URL) -> URL {
        workspace.appendingPathComponent(".localai/rag-index.json")
    }

    /// Carica un indice già costruito per questo workspace, se esiste.
    func loadIndexIfAvailable(workspace: URL?) {
        guard let workspace else { return }
        guard indexedWorkspace != workspace || chunks.isEmpty else { return }
        let url = Self.indexURL(for: workspace)
        guard let data = try? Data(contentsOf: url),
            let saved = try? JSONDecoder().decode([RAGChunk].self, from: data), !saved.isEmpty
        else { return }
        chunks = saved
        indexedWorkspace = workspace
        state = .ready(chunks: saved.count)
    }

    // MARK: - Costruzione indice

    func buildIndex(workspace: URL) {
        guard indexTask == nil else { return }
        indexTask = Task {
            defer { indexTask = nil }
            do {
                try await loadModelIfNeeded()

                // 1. raccogli i file di testo del workspace
                let files = Self.collectFiles(in: workspace)
                var pieces: [(file: String, text: String)] = []
                let root = workspace.standardizedFileURL.path
                for file in files {
                    guard let text = TrainingManager.extractText(file) else { continue }
                    var relative = file.standardizedFileURL.path
                    if relative.hasPrefix(root + "/") {
                        relative = String(relative.dropFirst(root.count + 1))
                    }
                    for piece in TrainingManager.split(text: text, size: 900) {
                        pieces.append((relative, piece))
                    }
                }
                guard !pieces.isEmpty else {
                    state = .failed("nessun documento indicizzabile nel workspace")
                    return
                }

                // 2. embedding a lotti (prefisso "passage:" richiesto da e5)
                var built: [RAGChunk] = []
                state = .indexing(done: 0, total: pieces.count)
                let batchSize = 16
                for start in stride(from: 0, to: pieces.count, by: batchSize) {
                    if Task.isCancelled { return }
                    let batch = Array(pieces[start ..< min(start + batchSize, pieces.count)])
                    let vectors = try await embed(texts: batch.map { "passage: " + $0.text })
                    for (piece, vector) in zip(batch, vectors) {
                        built.append(RAGChunk(file: piece.file, text: piece.text, vector: vector))
                    }
                    state = .indexing(done: built.count, total: pieces.count)
                }

                // 3. salva e attiva
                chunks = built
                indexedWorkspace = workspace
                let indexURL = Self.indexURL(for: workspace)
                try? FileManager.default.createDirectory(
                    at: indexURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                if let data = try? JSONEncoder().encode(built) {
                    try? data.write(to: indexURL)
                }
                state = .ready(chunks: built.count)
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func cancelIndexing() {
        indexTask?.cancel()
        indexTask = nil
        if case .indexing = state { state = .idle }
    }

    private static func collectFiles(in workspace: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: workspace,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        var files: [URL] = []
        for case let url as URL in enumerator {
            if excludedDirs.contains(url.lastPathComponent) {
                enumerator.skipDescendants()
                continue
            }
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                values.isRegularFile == true,
                (values.fileSize ?? 0) < 4_000_000,
                indexedExtensions.contains(url.pathExtension.lowercased())
            else { continue }
            files.append(url)
            if files.count >= 800 { break }
        }
        return files
    }

    // MARK: - Ricerca

    /// Ricerca semantica: restituisce i passaggi più pertinenti, formattati per il modello.
    func search(_ query: String, topK: Int = 5) async throws -> String {
        guard !chunks.isEmpty else {
            throw AgentToolbox.ToolboxError(
                "The semantic index is empty. Ask the user to build it from the RAG panel.")
        }
        try await loadModelIfNeeded()
        let queryVector = try await embed(texts: ["query: " + query])[0]

        let scored = chunks
            .map { chunk in (chunk, Self.dot(queryVector, chunk.vector)) }
            .sorted { $0.1 > $1.1 }
            .prefix(topK)

        return scored.enumerated()
            .map { index, item in
                let (chunk, score) = item
                return "[\(index + 1)] \(chunk.file) (rilevanza \(String(format: "%.2f", score)))\n\(chunk.text)"
            }
            .joined(separator: "\n\n---\n\n")
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var total: Float = 0
        for index in 0 ..< min(a.count, b.count) {
            total += a[index] * b[index]
        }
        return total
    }

    // MARK: - Embedding

    private func loadModelIfNeeded() async throws {
        guard container == nil else { return }
        state = .loadingModel(fraction: 0)
        container = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: Self.embedderConfiguration
        ) { @Sendable progress in
            let fraction = progress.fractionCompleted
            Task { @MainActor [weak self] in
                if case .loadingModel = self?.state {
                    self?.state = .loadingModel(fraction: fraction)
                }
            }
        }
    }

    /// Embedding normalizzati (coseno = prodotto scalare). Vedi README MLXEmbedders.
    private func embed(texts: [String]) async throws -> [[Float]] {
        guard let container else {
            throw AgentToolbox.ToolboxError("Embedding model not loaded.")
        }
        return try await container.perform { context in
            let inputs = texts.map { text in
                Array(context.tokenizer.encode(text: text, addSpecialTokens: true).prefix(510))
            }
            let eos = context.tokenizer.eosTokenId ?? 0
            let maxLength = inputs.reduce(16) { max($0, $1.count) }
            let padded = stacked(
                inputs.map { tokens in
                    MLXArray(tokens + Array(repeating: eos, count: maxLength - tokens.count))
                })
            let mask = padded .!= MLXArray(eos)
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.pooling(
                context.model(
                    padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                mask: mask,
                normalize: true,
                applyLayerNorm: false
            )
            output.eval()
            return output.map { $0.asArray(Float.self) }
        }
    }
}
