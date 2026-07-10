//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import QwenLocalCore
import Foundation
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Observation
import Tokenizers

/// Un messaggio della conversazione.
struct ChatMessage: Identifiable, Equatable {
    enum Role {
        case user
        case assistant
        /// Una chiamata a uno strumento fatta dal modello (modalità agente).
        case tool
    }

    let id = UUID()
    let role: Role
    var content: String = ""
    /// Contenuto del blocco <think>…</think> dei modelli Qwen3, mostrato a parte.
    var thinking: String = ""
    var tokensPerSecond: Double?
    var isComplete: Bool = false
    /// Nomi dei file allegati dall'utente (solo messaggi user).
    var attachments: [String] = []
    /// Percorso di un'immagine generata (solo messaggi assistant).
    var imagePath: String?

    // Solo per role == .tool
    var toolName: String = ""
    var toolArgs: String = ""
    var toolDenied: Bool = false
    var toolFailed: Bool = false
}

/// Richiesta di conferma per una tool call "pericolosa" (scrittura file o shell).
struct ApprovalRequest: Identifiable {
    let id = UUID()
    let toolName: String
    let detail: String
    let continuation: CheckedContinuation<Bool, Never>
}

/// Carica i modelli Qwen (download da Hugging Face + pesi in memoria via MLX)
/// e gestisce la conversazione — con streaming dei token e, in modalità agente,
/// il loop di tool calling (lettura/scrittura file, ricerca, shell).
@MainActor
@Observable
final class QwenEngine {

    enum Phase: Equatable {
        case idle
        case downloading(fraction: Double)
        case loadingWeights
        case ready
        case failed(String)

        var isBusy: Bool {
            switch self {
            case .downloading, .loadingWeights: true
            default: false
            }
        }
    }

    private(set) var phase: Phase = .idle
    private(set) var activeModel: CatalogModel?
    private(set) var isGenerating = false
    private(set) var messages: [ChatMessage] = []

    // Impostazioni: valgono dalla prossima nuova chat (la sessione mantiene la KV cache).
    var systemPrompt: String = ""
    var temperature: Double = 0.7
    var enableThinking: Bool = true

    // Limiti memoria/GPU: si applicano subito (anche a modello caricato).
    var gpuCacheLimitMB: Double = 64 { didSet { applyGPULimits() } }
    var gpuMemoryLimitGB: Double = QwenEngine.defaultMemoryLimitGB { didSet { applyGPULimits() } }
    private(set) var gpuActiveBytes = 0
    private(set) var gpuPeakBytes = 0

    static var physicalRAMGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
    }
    /// Default prudente: 60% della RAM fisica come tetto rigido per la GPU.
    static var defaultMemoryLimitGB: Double {
        (physicalRAMGB * 0.6).rounded()
    }

    // Memoria di progetto (file .md stile CLAUDE.md) e tracking del contesto
    private(set) var projectMemory: String?
    private(set) var projectMemoryFileName: String?
    private(set) var contextTokens = 0
    private var compactedSummary: String?
    /// Finestra di contesto nativa dei modelli Qwen3 (32k token).
    static let contextWindow = 32_768
    static let projectMemoryFileNames = ["LOCALAI.md", "CLAUDE.md", "AGENTS.md"]

    /// Ricerca semantica RAG fornita dall'esterno (RAGManager); il tool
    /// search_documents appare alla prossima nuova chat.
    var semanticSearchProvider: (@Sendable (String) async throws -> String)?

    // Modalità agente
    var agentMode: Bool = true
    var autoApprove: Bool = false
    private(set) var workspaceURL: URL?
    /// Se impostata, l'agente può scrivere SOLO in questa cartella (dentro il workspace).
    private(set) var writeBoundaryURL: URL?
    var pendingApproval: ApprovalRequest?

    private var container: ModelContainer?
    private var session: ChatSession?
    private var toolbox: AgentToolbox?
    private var loadTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?
    /// Accumulo grezzo del testo del segmento assistant corrente (tra una tool call e l'altra).
    private var currentRaw = ""
    /// Tool call eseguite nel turno corrente: oltre la soglia la generazione viene fermata,
    /// per evitare che un modello piccolo entri in un loop infinito di chiamate.
    private var toolCallsThisTurn = 0
    private static let maxToolCallsPerTurn = 30

    var isAgentActive: Bool {
        agentMode && workspaceURL != nil
    }

    init() {
        applyGPULimits()
    }

    /// Applica i limiti alla GPU: cache dei buffer e tetto rigido di memoria.
    /// Con `relaxed: false` un'allocazione oltre il limite fallisce con un errore
    /// (mostrato in chat) invece di saturare la RAM del sistema.
    private func applyGPULimits() {
        GPU.set(cacheLimit: Int(gpuCacheLimitMB * 1_048_576))
        GPU.set(memoryLimit: Int(gpuMemoryLimitGB * 1_073_741_824), relaxed: false)
    }

    func refreshGPUStats() {
        gpuActiveBytes = GPU.activeMemory
        gpuPeakBytes = GPU.peakMemory
    }

    // MARK: - Caricamento modello

    /// Scarica (se serve) e carica in memoria il modello scelto.
    func load(_ model: CatalogModel, onDownloadFinished: @escaping @MainActor () -> Void = {}) {
        stopGeneration()
        loadTask?.cancel()
        unload()
        activeModel = model
        phase = .downloading(fraction: 0)

        loadTask = Task {
            do {
                let configuration: ModelConfiguration
                if model.id.hasPrefix("local:") {
                    configuration = ModelConfiguration(
                        directory: URL(fileURLWithPath: String(model.id.dropFirst("local:".count))))
                } else if model.id.localizedCaseInsensitiveContains("glm") {
                    // La famiglia GLM emette le tool call nel proprio formato.
                    configuration = ModelConfiguration(id: model.id, toolCallFormat: .glm4)
                } else {
                    configuration = ModelConfiguration(id: model.id)
                }
                let loaded = try await #huggingFaceLoadModelContainer(
                    configuration: configuration
                ) { @Sendable progress in
                    let fraction = progress.fractionCompleted
                    Task { @MainActor [weak self] in
                        guard let self, self.activeModel?.id == model.id else { return }
                        self.phase =
                            fraction >= 1 ? .loadingWeights : .downloading(fraction: fraction)
                    }
                }
                guard !Task.isCancelled, activeModel?.id == model.id else { return }
                applyGPULimits()
                container = loaded
                phase = .ready
                refreshGPUStats()
                startNewChat()
                onDownloadFinished()
            } catch is CancellationError {
                // caricamento annullato: nessun errore da mostrare
            } catch {
                guard activeModel?.id == model.id else { return }
                phase = .failed(error.localizedDescription)
                activeModel = nil
            }
        }
    }

    /// Scarica il modello dalla memoria (i file su disco restano).
    func unload() {
        stopGeneration()
        session = nil
        container = nil
        messages = []
        activeModel = nil
        phase = .idle
        GPU.clearCache()
        GPU.resetPeakMemory()
        refreshGPUStats()
    }

    /// Imposta la cartella di lavoro dell'agente.
    func setWorkspace(_ url: URL?) {
        workspaceURL = url
        writeBoundaryURL = nil  // il confine appartiene al workspace precedente
        loadProjectMemory()
        if phase == .ready {
            startNewChat()
        }
    }

    /// Limita le scritture dell'agente a una sottocartella del workspace (nil = tutto il workspace).
    func setWriteBoundary(_ url: URL?) {
        writeBoundaryURL = url
        if phase == .ready {
            startNewChat()
        }
    }

    /// Legge il file di memoria del progetto (LOCALAI.md / CLAUDE.md / AGENTS.md)
    /// dalla radice del workspace. Viene iniettato nelle istruzioni di sistema.
    private func loadProjectMemory() {
        projectMemory = nil
        projectMemoryFileName = nil
        guard let workspace = workspaceURL else { return }
        for name in Self.projectMemoryFileNames {
            let url = workspace.appendingPathComponent(name)
            guard let text = try? String(contentsOf: url, encoding: .utf8),
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { continue }
            // Cap prudente: un file di memoria enorme mangerebbe la finestra di contesto.
            projectMemory = String(text.prefix(24_000))
            projectMemoryFileName = name
            return
        }
    }

    /// Carica un modello MLX da una cartella locale (es. l'output di mlx_lm.fuse).
    func loadLocalModel(directory: URL) {
        let model = CatalogModel(
            id: "local:" + directory.path,
            displayName: directory.lastPathComponent,
            sizeGB: 0,
            supportsThinking: true,
            note: "modello locale (fine-tuned)")
        load(model)
    }

    /// Sessione usa-e-getta sul modello caricato, per lavori di servizio
    /// (es. generazione del dataset di training). Nessun tool, niente storia.
    func utilitySession(instructions: String? = nil) -> ChatSession? {
        guard let container, phase == .ready else { return nil }
        return ChatSession(
            container,
            instructions: instructions,
            generateParameters: GenerateParameters(maxTokens: 2048, temperature: 0.3),
            additionalContext: ["enable_thinking": false])
    }

    // MARK: - Conversazione

    /// Nuova chat: azzera i messaggi e ricrea la sessione con le impostazioni correnti.
    func startNewChat() {
        compactedSummary = nil
        startSession()
    }

    private func startSession() {
        stopGeneration()
        messages = []
        currentRaw = ""
        contextTokens = 0
        guard let container else { return }

        loadProjectMemory()
        var instructions = ""
        if isAgentActive, let workspace = workspaceURL {
            toolbox = AgentToolbox(
                workspace: workspace,
                writeBoundary: writeBoundaryURL,
                semanticSearch: semanticSearchProvider)
            instructions = Self.agentInstructions(
                workspace: workspace,
                writeBoundary: writeBoundaryURL,
                hasSemanticSearch: semanticSearchProvider != nil)
        } else {
            toolbox = nil
        }
        if let memory = projectMemory, let name = projectMemoryFileName {
            instructions += "\n\n## Project notes from \(name) — follow these\n\(memory)"
        }
        if let summary = compactedSummary {
            instructions += "\n\n## Summary of the previous conversation (context was compacted)\n\(summary)"
        }
        let userPrompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !userPrompt.isEmpty {
            instructions += instructions.isEmpty ? userPrompt : "\n\n" + userPrompt
        }

        var toolDispatch: (@Sendable (ToolCall) async throws -> String)?
        if toolbox != nil {
            toolDispatch = { @Sendable [weak self] call in
                await self?.dispatch(call)
                    ?? "The session was closed. Stop and report this to the user."
            }
        }

        session = ChatSession(
            container,
            instructions: instructions.isEmpty ? nil : instructions,
            generateParameters: GenerateParameters(maxTokens: 4096, temperature: Float(temperature)),
            additionalContext: ["enable_thinking": enableThinking],
            tools: toolbox.map { $0.specs },
            toolDispatch: toolDispatch
        )
    }

    func send(_ prompt: String, attachments: [URL] = []) {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty || !attachments.isEmpty else { return }
        guard let session, phase == .ready, !isGenerating else { return }

        let visibleText = prompt.isEmpty
            ? "Analizza il contenuto dei file allegati." : prompt
        var userMessage = ChatMessage(role: .user, content: visibleText, isComplete: true)
        userMessage.attachments = attachments.map(\.lastPathComponent)
        messages.append(userMessage)

        // Il modello riceve il testo + il contenuto estratto dagli allegati.
        var modelPrompt = visibleText
        for url in attachments {
            let name = url.lastPathComponent
            guard let text = TrainingManager.extractText(url),
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                modelPrompt += "\n\n[Attached file \(name): unreadable or empty]"
                continue
            }
            let capped = text.count > 12_000
                ? String(text.prefix(12_000)) + "\n…[file truncated: \(text.count) characters total]"
                : text
            modelPrompt += "\n\n## Attached file: \(name)\n```\n\(capped)\n```"
        }

        currentRaw = ""
        toolCallsThisTurn = 0
        messages.append(ChatMessage(role: .assistant))
        isGenerating = true

        let finalPrompt = modelPrompt
        generationTask = Task {
            do {
                for try await item in session.streamDetails(to: finalPrompt) {
                    if let chunk = item.chunk {
                        appendChunk(chunk)
                    }
                    if let info = item.info {
                        updateLastAssistantMessage { $0.tokensPerSecond = info.tokensPerSecond }
                        // Con la KV cache promptTokenCount copre solo il segmento nuovo:
                        // la somma cumulativa approssima il contesto totale occupato.
                        contextTokens += info.promptTokenCount + info.generationTokenCount
                        refreshGPUStats()
                    }
                }
            } catch is CancellationError {
                // generazione interrotta dall'utente
            } catch {
                appendChunk("")
                updateLastAssistantMessage { message in
                    if message.content.isEmpty {
                        message.content = "⚠️ Errore: \(error.localizedDescription)"
                    }
                }
            }
            finishStreaming()
        }
    }

    /// Compatta il contesto (come /compact di Claude Code): chiede al modello un
    /// riassunto della conversazione e riparte con una sessione fresca che lo contiene.
    func compactContext() {
        guard let session, phase == .ready, !isGenerating, !messages.isEmpty else { return }
        isGenerating = true
        messages.append(ChatMessage(role: .assistant, content: "✻ compatto il contesto…"))
        generationTask = Task {
            do {
                let summary = try await session.respond(
                    to: """
                    Summarize this conversation for a hand-off: user goals, decisions taken,                     files read or modified (with paths), current state, and open next steps.                     Use concise bullet points, in the user's language. No preamble.
                    """)
                compactedSummary = summary
                isGenerating = false
                startSession()
                messages = [ChatMessage(
                    role: .assistant,
                    content: "✻ contesto compattato — riassunto riportato nella nuova sessione:\n\n" + summary,
                    isComplete: true)]
            } catch {
                isGenerating = false
                updateLastAssistantMessage { message in
                    message.content = "⚠️ compattazione fallita: \(error.localizedDescription)"
                    message.isComplete = true
                }
            }
        }
    }

    /// Fa generare all'agente il file di memoria del progetto (come /init di Claude Code).
    func generateProjectMemory() {
        guard isAgentActive else { return }
        send("""
            Explore this project: call list_directory on the root, then read the most             informative files (README, package manifests, main sources) — at most 8 tool             calls. Then use write_file to create LOCALAI.md in the workspace root with:             a one-paragraph project overview, the directory structure, build/run/test             commands, and code conventions. Write the file in Italian. Keep it under 60             lines. Confirm briefly when done.
            """)
    }

    /// Aggiunge in transcript un messaggio utente e una nota dell'assistente
    /// senza passare dal modello (es. suggerimenti quando nessun LLM è caricato).
    func appendNote(user: String?, note: String) {
        if let user {
            messages.append(ChatMessage(role: .user, content: user, isComplete: true))
        }
        messages.append(ChatMessage(role: .assistant, content: note, isComplete: true))
    }

    /// Registra in chat una richiesta di generazione immagine; restituisce
    /// l'id del segnaposto assistant da completare a generazione finita.
    func beginImageGeneration(prompt: String) -> UUID {
        messages.append(ChatMessage(role: .user, content: "/img " + prompt, isComplete: true))
        var placeholder = ChatMessage(role: .assistant, content: "✻ genero l'immagine…")
        messages.append(placeholder)
        return placeholder.id
    }

    func completeImageGeneration(id: UUID, imagePath: String?, error: String?) {
        guard let index = messages.lastIndex(where: { $0.id == id }) else { return }
        if let imagePath {
            messages[index].content = ""
            messages[index].imagePath = imagePath
        } else {
            messages[index].content = "⚠️ generazione fallita: \(error ?? "errore sconosciuto")"
        }
        messages[index].isComplete = true
    }

    func updateImageProgress(id: UUID, step: Int, total: Int) {
        guard let index = messages.lastIndex(where: { $0.id == id }) else { return }
        messages[index].content = total > 0
            ? "✻ genero l'immagine… passo \(step)/\(total)"
            : "✻ genero l'immagine…"
    }

    func stopGeneration() {
        resolveApproval(false)
        generationTask?.cancel()
        generationTask = nil
        finishStreaming()
    }

    /// Risposta dell'utente alla richiesta di approvazione corrente.
    func resolveApproval(_ approved: Bool) {
        pendingApproval?.continuation.resume(returning: approved)
        pendingApproval = nil
    }

    // MARK: - Streaming interno

    private func appendChunk(_ chunk: String) {
        // Dopo una tool call il testo riparte in una nuova bolla assistant.
        if messages.last?.role != .assistant || messages.last?.isComplete == true {
            messages.append(ChatMessage(role: .assistant))
            currentRaw = ""
        }
        currentRaw += chunk
        let (thinking, content) = Self.splitThinking(currentRaw)
        updateLastAssistantMessage { message in
            message.thinking = thinking
            message.content = content
        }
    }

    private func finishStreaming() {
        isGenerating = false
        refreshGPUStats()
        // Chiude l'ultima bolla; se è rimasta vuota la rimuove.
        if let last = messages.last, last.role == .assistant {
            if last.content.isEmpty, last.thinking.isEmpty {
                messages.removeLast()
            } else {
                updateLastAssistantMessage { $0.isComplete = true }
            }
        }
    }

    private func updateLastAssistantMessage(_ mutate: (inout ChatMessage) -> Void) {
        guard let index = messages.lastIndex(where: { $0.role == .assistant }) else { return }
        mutate(&messages[index])
    }

    // MARK: - Dispatch delle tool call

    private func dispatch(_ call: ToolCall) async -> String {
        // Il testo generato prima della tool call diventa una bolla conclusa.
        if let last = messages.last, last.role == .assistant {
            if last.content.isEmpty, last.thinking.isEmpty {
                messages.removeLast()
            } else {
                updateLastAssistantMessage { $0.isComplete = true }
            }
        }
        currentRaw = ""

        toolCallsThisTurn += 1
        if toolCallsThisTurn > Self.maxToolCallsPerTurn {
            var limitMessage = ChatMessage(role: .tool)
            limitMessage.toolName = call.function.name
            limitMessage.toolFailed = true
            limitMessage.content = "Limite di \(Self.maxToolCallsPerTurn) tool call per turno raggiunto: generazione interrotta."
            limitMessage.isComplete = true
            messages.append(limitMessage)
            generationTask?.cancel()
            return "Tool call limit reached. Stop now and summarize what you did for the user."
        }

        var toolMessage = ChatMessage(role: .tool)
        toolMessage.toolName = call.function.name
        toolMessage.toolArgs = Self.formatArguments(call)
        messages.append(toolMessage)
        let messageID = toolMessage.id

        guard let toolbox else {
            updateToolMessage(messageID) { $0.toolFailed = true; $0.isComplete = true }
            return "No workspace is configured. Tell the user to select a workspace folder."
        }

        if AgentToolbox.toolsNeedingApproval.contains(call.function.name), !autoApprove {
            let approved = await withCheckedContinuation { continuation in
                pendingApproval = ApprovalRequest(
                    toolName: call.function.name,
                    detail: Self.formatArguments(call),
                    continuation: continuation
                )
            }
            if !approved {
                updateToolMessage(messageID) {
                    $0.toolDenied = true
                    $0.content = "Negato dall'utente"
                    $0.isComplete = true
                }
                return "The user DENIED this tool call. Do not retry it. Ask the user how to proceed."
            }
        }

        do {
            let result = try await toolbox.dispatch(call)
            updateToolMessage(messageID) {
                $0.content = result
                $0.isComplete = true
            }
            return result
        } catch {
            updateToolMessage(messageID) {
                $0.toolFailed = true
                $0.content = error.localizedDescription
                $0.isComplete = true
            }
            return "Tool error: \(error.localizedDescription)"
        }
    }

    private func updateToolMessage(_ id: UUID, _ mutate: (inout ChatMessage) -> Void) {
        guard let index = messages.lastIndex(where: { $0.id == id }) else { return }
        mutate(&messages[index])
    }

    private static func formatArguments(_ call: ToolCall) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(call.function.arguments),
            let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    private static func agentInstructions(
        workspace: URL, writeBoundary: URL? = nil, hasSemanticSearch: Bool
    ) -> String {
        """
        You are a coding agent working inside this workspace directory: \(workspace.path)

        You have tools to explore and modify the project: read_file, write_file, edit_file, \
        list_directory, glob, grep, and bash (shell commands run with the workspace as cwd).\
        \(hasSemanticSearch
            ? "\nYou also have search_documents: semantic search over the indexed workspace documents — prefer it over grep for questions about document CONTENT or meaning; use grep for exact identifiers."
            : "")

        Guidelines:
        - All file paths are relative to the workspace root.
        - Explore before you act: use list_directory, glob and grep to find the right files, \
        and read_file before editing.
        - Prefer edit_file for small, targeted changes; use write_file only for new files or \
        full rewrites.
        - After changing code, verify it when possible (e.g. run a build or tests via bash).
        - If a tool call is denied by the user, do not retry it; ask how to proceed.\
        \(writeBoundary.map { "\n- IMPORTANT: you may WRITE files only inside '\($0.lastPathComponent)/'. Reading is allowed everywhere in the workspace." } ?? "")
        - Keep your text responses short and factual. Answer in the user's language.
        """
    }

    /// Separa il blocco di ragionamento dalla risposta. Gestisce entrambi gli stili:
    /// Qwen3 emette `<think>…</think>`, GLM apre il blocco nel template e
    /// genera solo `ragionamento…</think>risposta` (chiusura senza apertura).
    static func splitThinking(_ raw: String) -> (thinking: String, content: String) {
        guard let start = raw.range(of: "<think>") else {
            if let end = raw.range(of: "</think>") {
                let thinking = String(raw[..<end.lowerBound])
                let content = String(raw[end.upperBound...])
                return (
                    thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                    content.trimmingCharacters(in: .whitespacesAndNewlines)
                )
            }
            return ("", raw.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let end = raw.range(of: "</think>", range: start.upperBound..<raw.endIndex) else {
            // Il modello sta ancora "pensando".
            let thinking = String(raw[start.upperBound...])
            return (thinking.trimmingCharacters(in: .whitespacesAndNewlines), "")
        }
        let thinking = String(raw[start.upperBound..<end.lowerBound])
        let content = String(raw[end.upperBound...])
        return (
            thinking.trimmingCharacters(in: .whitespacesAndNewlines),
            content.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }
}