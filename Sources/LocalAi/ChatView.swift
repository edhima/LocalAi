import QwenLocalCore
import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// La vista principale della conversazione, in stile terminale Claude Code:
/// tema scuro, font mono, transcript piatto con `>` per l'utente e `⏺` per i tool.
struct ChatView: View {
    @Bindable var engine: QwenEngine
    @Bindable var rag: RAGManager
    @Bindable var imageGen: ImageGenManager
    @Environment(\.openWindow) private var openWindow
    @State private var showRAG = false
    @State private var attachedFiles: [URL] = []
    @State private var draft = ""
    @State private var showSettings = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            switch engine.phase {
            case .idle:
                // Col solo modello immagine montato la chat serve comunque (/img).
                if imageGen.isReady {
                    transcript
                } else {
                    centered {
                        VStack(spacing: 10) {
                            Text("✻")
                                .font(Theme.mono(40))
                                .foregroundStyle(Theme.accent)
                            Text("Nessun modello caricato")
                                .font(Theme.mono(15, weight: .semibold))
                                .foregroundStyle(Theme.text)
                            Text("Scegli un modello Qwen dalla barra laterale.\nAl primo avvio verrà scaricato da Hugging Face.")
                                .font(Theme.mono(12))
                                .foregroundStyle(Theme.secondary)
                                .multilineTextAlignment(.center)
                        }
                    }
                }
            case .downloading(let fraction):
                centered {
                    VStack(spacing: 12) {
                        Text("✻ download di \(engine.activeModel?.shortName ?? "modello")…")
                            .font(Theme.mono(13))
                            .foregroundStyle(Theme.accent)
                        ProgressView(value: fraction)
                            .frame(maxWidth: 340)
                            .tint(Theme.accent)
                        Text("\(Int(fraction * 100))%")
                            .font(Theme.mono(13))
                            .foregroundStyle(Theme.secondary)
                    }
                }
            case .loadingWeights:
                centered {
                    VStack(spacing: 10) {
                        ProgressView().tint(Theme.accent)
                        Text("carico i pesi in memoria…")
                            .font(Theme.mono(13))
                            .foregroundStyle(Theme.secondary)
                    }
                }
            case .failed(let message):
                centered {
                    VStack(spacing: 10) {
                        Text("✗ errore di caricamento")
                            .font(Theme.mono(14, weight: .semibold))
                            .foregroundStyle(Theme.red)
                        Text(message)
                            .font(Theme.mono(12))
                            .foregroundStyle(Theme.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 480)
                    }
                }
            case .ready:
                transcript
                    .dropDestination(for: URL.self) { urls, _ in
                        let files = urls.filter {
                            !$0.hasDirectoryPath && !attachedFiles.contains($0)
                        }
                        attachedFiles.append(contentsOf: files)
                        return !files.isEmpty
                    }
            }

            if let approval = engine.pendingApproval {
                ApprovalPanel(request: approval) { engine.resolveApproval($0) }
            }

            if engine.phase == .ready || (engine.phase == .idle && imageGen.isReady) {
                inputBox
                statusBar
            }
        }
        .background(Theme.background)
        .navigationTitle("LocalAi")
        .navigationSubtitle(engine.activeModel?.shortName ?? "")
        .toolbar {
            ToolbarItem {
                Button(action: chooseWorkspace) {
                    Label(
                        engine.workspaceURL?.lastPathComponent ?? "Workspace…",
                        systemImage: "folder"
                    )
                    .labelStyle(.titleAndIcon)
                }
                .help("Scegli la cartella di progetto per la modalità agente")
            }
            ToolbarItem {
                Button {
                    showRAG.toggle()
                } label: {
                    Label("RAG", systemImage: "sparkle.magnifyingglass")
                }
                .popover(isPresented: $showRAG, arrowEdge: .bottom) {
                    RAGPane(rag: rag, engine: engine)
                }
                .help("Ricerca semantica sui documenti del workspace")
            }
            ToolbarItem {
                Button {
                    openWindow(id: "training")
                } label: {
                    Label("Training", systemImage: "graduationcap")
                }
                .help("Fine-tuning LoRA: prepara i dati e addestra un Qwen personalizzato")
            }
            ToolbarItem {
                Button {
                    engine.compactContext()
                } label: {
                    Label("Compatta", systemImage: "arrow.down.right.and.arrow.up.left")
                }
                .disabled(engine.phase != .ready || engine.isGenerating || engine.messages.isEmpty)
                .help("Compatta il contesto: riassume la conversazione e riparte dal riassunto")
            }
            ToolbarItem {
                Button {
                    engine.startNewChat()
                } label: {
                    Label("Nuova chat", systemImage: "square.and.pencil")
                }
                .disabled(engine.phase != .ready)
                .help("Nuova chat (applica anche le impostazioni)")
            }
            ToolbarItem {
                Button {
                    showSettings.toggle()
                } label: {
                    Label("Impostazioni", systemImage: "slider.horizontal.3")
                }
                .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                    SettingsPane(engine: engine)
                }
            }
        }
        .onAppear { syncRAG() }
        .onChange(of: imageGen.state) {
            // aggiorna il segnaposto "genero…" con il passo corrente
            if case .generating(let step, let total) = imageGen.state,
                let last = engine.messages.last(where: { $0.role == .assistant && !$0.isComplete }) {
                engine.updateImageProgress(id: last.id, step: step, total: total)
            }
        }
        .onChange(of: engine.workspaceURL) { syncRAG() }
        .onChange(of: rag.isReady) { syncRAG() }
        // Esc interrompe la generazione (quando non c'è un'approvazione in corso,
        // che gestisce Esc per conto suo come "Nega").
        .background {
            Button("") {
                engine.stopGeneration()
            }
            .keyboardShortcut(.cancelAction)
            .opacity(0)
            .disabled(!engine.isGenerating || engine.pendingApproval != nil)
        }
    }

    private func centered<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack { content() }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
    }

    /// Collega il tool search_documents quando l'indice è valido per il workspace corrente.
    private func syncRAG() {
        rag.loadIndexIfAvailable(workspace: engine.workspaceURL)
        let valid = rag.isReady && rag.indexedWorkspace == engine.workspaceURL
        let ragRef = rag
        var provider: (@Sendable (String) async throws -> String)?
        if valid {
            provider = { query in try await ragRef.search(query) }
        }
        engine.semanticSearchProvider = provider
        // Sessione vuota: la ricrea in silenzio così il tool è subito disponibile.
        if engine.phase == .ready, engine.messages.isEmpty {
            engine.startNewChat()
        }
    }

    private func chooseWorkspace() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli la cartella di progetto in cui l'agente potrà lavorare"
        panel.prompt = "Usa come workspace"
        if panel.runModal() == .OK {
            engine.setWorkspace(panel.url)
        }
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if engine.messages.isEmpty {
                        WelcomeBanner(engine: engine, imageGen: imageGen)
                            .padding(.top, 24)
                    }
                    ForEach(engine.messages) { message in
                        TranscriptRow(message: message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 14)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: engine.messages.last?.content) {
                if let last = engine.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .onChange(of: engine.messages.count) {
                if let last = engine.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    // MARK: - Input

    private var inputBox: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !attachedFiles.isEmpty {
                HStack(spacing: 6) {
                    ForEach(attachedFiles, id: \.self) { file in
                        HStack(spacing: 4) {
                            Image(systemName: "paperclip")
                                .font(.system(size: 9))
                            Text(file.lastPathComponent)
                                .font(Theme.mono(10))
                                .lineLimit(1)
                            Button {
                                attachedFiles.removeAll { $0 == file }
                            } label: {
                                Image(systemName: "xmark").font(.system(size: 8))
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Theme.panel, in: Capsule())
                        .foregroundStyle(Theme.secondary)
                    }
                    Spacer()
                }
            }

            HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(">")
                .font(Theme.mono(14, weight: .bold))
                .foregroundStyle(Theme.accent)

            TextField(
                engine.phase != .ready && imageGen.isReady
                    ? "/img descrizione dell'immagine da generare…"
                    : (engine.isAgentActive
                        ? "chiedi qualcosa sul progetto o fai fare una modifica…"
                        : "scrivi un messaggio…"),
                text: $draft, axis: .vertical
            )
            .textFieldStyle(.plain)
            .font(Theme.mono(13))
            .foregroundStyle(Theme.text)
            .lineLimit(1...8)
            .focused($inputFocused)
            .onSubmit(sendDraft)
            .disabled(engine.pendingApproval != nil)

            Button(action: pickAttachments) {
                Image(systemName: "paperclip")
                    .foregroundStyle(Theme.dim)
            }
            .buttonStyle(.borderless)
            .help("Allega file da analizzare (txt, md, csv, json, pdf) — o trascinali qui")
            .disabled(engine.isGenerating)

            if engine.isGenerating {
                Button(action: engine.stopGeneration) {
                    Image(systemName: "stop.fill")
                        .foregroundStyle(Theme.red)
                }
                .buttonStyle(.borderless)
                .help("Interrompi (esc)")
            }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(inputFocused ? Theme.accent.opacity(0.6) : Theme.border, lineWidth: 1)
        )
        .padding(.horizontal, 14)
        .padding(.top, 8)
        .task { inputFocused = true }
    }

    private var statusBar: some View {
        HStack(spacing: 6) {
            if engine.isGenerating {
                Text("✻ generazione…")
                    .foregroundStyle(Theme.accent)
                Text("(esc per interrompere)")
                    .foregroundStyle(Theme.dim)
            } else if let workspace = engine.workspaceURL, engine.isAgentActive {
                Text("⏺")
                    .foregroundStyle(Theme.green)
                Text(workspace.path.replacingOccurrences(
                    of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~"))
                    .foregroundStyle(Theme.dim)
                    .lineLimit(1)
                    .truncationMode(.middle)
            } else {
                Text("nessun workspace — solo chat")
                    .foregroundStyle(Theme.dim)
            }

            Spacer()

            if engine.contextTokens > 0 {
                let nearFull = engine.contextTokens > QwenEngine.contextWindow * 85 / 100
                Text(String(format: "ctx %.1fk/%dk", Double(engine.contextTokens) / 1000,
                            QwenEngine.contextWindow / 1024))
                    .foregroundStyle(nearFull ? Theme.orange : Theme.dim)
                if nearFull {
                    Text("→ usa Compatta")
                        .foregroundStyle(Theme.orange)
                }
            }
            if engine.gpuActiveBytes > 0 {
                Text(String(
                    format: "mem %.1fG · picco %.1fG",
                    Double(engine.gpuActiveBytes) / 1_073_741_824,
                    Double(engine.gpuPeakBytes) / 1_073_741_824))
                    .foregroundStyle(Theme.dim)
            }
            if let tps = engine.messages.last(where: { $0.tokensPerSecond != nil })?.tokensPerSecond {
                Text(String(format: "%.1f tok/s", tps))
                    .foregroundStyle(Theme.dim)
            }
            Text(engine.activeModel?.shortName ?? "")
                .foregroundStyle(Theme.dim)
        }
        .font(Theme.mono(11))
        .padding(.horizontal, 16)
        .padding(.top, 5)
        .padding(.bottom, 8)
    }

    private func sendDraft() {
        let text = draft
        let files = attachedFiles
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty || !files.isEmpty else { return }

        // /img <prompt> → generazione immagine col checkpoint montato
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if trimmed.lowercased().hasPrefix("/img ") {
            let prompt = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            guard !prompt.isEmpty else { return }
            draft = ""
            guard imageGen.isReady else {
                let id = engine.beginImageGeneration(prompt: prompt)
                engine.completeImageGeneration(
                    id: id, imagePath: nil,
                    error: "nessun modello immagine montato (sidebar → modello immagine)")
                return
            }
            let id = engine.beginImageGeneration(prompt: prompt)
            imageGen.generate(prompt: prompt) { result in
                switch result {
                case .success(let url):
                    engine.completeImageGeneration(id: id, imagePath: url.path, error: nil)
                case .failure(let error):
                    engine.completeImageGeneration(
                        id: id, imagePath: nil, error: error.localizedDescription)
                }
            }
            return
        }

        guard engine.phase == .ready else {
            draft = ""
            engine.appendNote(
                user: trimmed,
                note: "⚠️ nessun modello chat caricato: per conversare scegli un Qwen dalla sidebar. Con il modello immagine montato puoi generare con /img <descrizione>.")
            return
        }

        draft = ""
        attachedFiles = []
        engine.send(text, attachments: files)
    }

    private func pickAttachments() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        var types: [UTType] = [.plainText, .utf8PlainText, .json, .pdf, .commaSeparatedText]
        if let markdown = UTType(filenameExtension: "md") { types.append(markdown) }
        panel.allowedContentTypes = types
        if panel.runModal() == .OK {
            attachedFiles.append(contentsOf: panel.urls.filter { !attachedFiles.contains($0) })
        }
    }
}

// MARK: - Banner di benvenuto

struct WelcomeBanner: View {
    var engine: QwenEngine
    var imageGen: ImageGenManager? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("✻")
                    .foregroundStyle(Theme.accent)
                    .font(Theme.mono(14, weight: .bold))
                Text("Benvenuto in LocalAi")
                    .font(Theme.mono(13, weight: .semibold))
                    .foregroundStyle(Theme.text)
            }
            .padding(.bottom, 4)

            infoLine("modello", engine.activeModel?.shortName ?? "— (scegli dalla sidebar)")
            if let imageGen, imageGen.isReady, case .ready(let model) = imageGen.state {
                infoLine("immagini", "\(model) — genera con /img <descrizione>")
            }
            infoLine(
                "workspace",
                engine.workspaceURL.map {
                    $0.path.replacingOccurrences(
                        of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~")
                } ?? "nessuno (scegli “Workspace…” dalla toolbar)")
            infoLine("agente", engine.isAgentActive ? "attivo — 7 tool disponibili" : "disattivo")
            infoLine(
                "rag",
                engine.semanticSearchProvider != nil
                    ? "indice attivo — tool search_documents disponibile"
                    : "non indicizzato (pannello RAG in toolbar)")
            infoLine(
                "memoria",
                engine.projectMemoryFileName.map { "\($0) ✓ caricato nel contesto" }
                    ?? (engine.isAgentActive ? "nessun file .md di progetto" : "—"))

            if engine.isAgentActive, engine.projectMemoryFileName == nil {
                Button {
                    engine.generateProjectMemory()
                } label: {
                    Text("✻ genera LOCALAI.md (l'agente esplora il progetto)")
                        .font(Theme.mono(12))
                        .foregroundStyle(Theme.accent)
                }
                .buttonStyle(.plain)
                .padding(.top, 4)
            }

            if engine.isAgentActive {
                Text("\nprova: > trova tutti i TODO nel progetto")
                    .font(Theme.mono(12))
                    .foregroundStyle(Theme.dim)
            }
        }
        .padding(14)
        .frame(maxWidth: 560, alignment: .leading)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Theme.accent.opacity(0.5), lineWidth: 1)
        )
    }

    private func infoLine(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(key.padding(toLength: 9, withPad: " ", startingAt: 0))
                .font(Theme.mono(12))
                .foregroundStyle(Theme.dim)
            Text(value)
                .font(Theme.mono(12))
                .foregroundStyle(Theme.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }
}

// MARK: - Righe del transcript

struct TranscriptRow: View {
    let message: ChatMessage

    var body: some View {
        switch message.role {
        case .user:
            UserRow(message: message)
        case .assistant:
            AssistantRow(message: message)
        case .tool:
            ToolRow(message: message)
        }
    }
}

struct UserRow: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(">")
                .font(Theme.mono(13, weight: .bold))
                .foregroundStyle(Theme.dim)
            VStack(alignment: .leading, spacing: 3) {
                Text(message.content)
                    .font(Theme.mono(13))
                    .foregroundStyle(Theme.secondary)
                    .textSelection(.enabled)
                ForEach(message.attachments, id: \.self) { name in
                    Label(name, systemImage: "paperclip")
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.dim)
                }
            }
        }
        .padding(.top, 6)
    }
}

struct AssistantRow: View {
    let message: ChatMessage
    @State private var showThinking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !message.thinking.isEmpty {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { showThinking.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Text("✻")
                            .foregroundStyle(Theme.accent.opacity(0.7))
                        Text(
                            message.content.isEmpty && !message.isComplete
                                ? "sto ragionando…" : "ragionamento")
                            .italic()
                        Image(systemName: showThinking ? "chevron.down" : "chevron.right")
                            .font(.system(size: 8))
                    }
                    .font(Theme.mono(12))
                    .foregroundStyle(Theme.dim)
                }
                .buttonStyle(.plain)

                if showThinking {
                    Text(message.thinking)
                        .font(Theme.mono(12))
                        .italic()
                        .foregroundStyle(Theme.dim)
                        .textSelection(.enabled)
                        .padding(.leading, 18)
                }
            }

            if let imagePath = message.imagePath {
                GeneratedImageView(path: imagePath)
            } else if message.content.isEmpty && !message.isComplete && message.thinking.isEmpty {
                Text("✻ …")
                    .font(Theme.mono(13))
                    .foregroundStyle(Theme.dim)
            } else if !message.content.isEmpty {
                ForEach(Array(Self.segments(from: message.content).enumerated()), id: \.offset) { _, segment in
                    switch segment {
                    case .text(let text):
                        Text(text)
                            .font(Theme.mono(13))
                            .foregroundStyle(Theme.text)
                            .textSelection(.enabled)
                            .lineSpacing(3)
                    case .code(let language, let body):
                        CodeBlockView(language: language, body: body)
                    }
                }
            }
        }
        .padding(.top, 2)
    }
}

struct ToolRow: View {
    let message: ChatMessage
    @State private var expanded = false

    private var dotColor: Color {
        if message.toolDenied { return Theme.orange }
        if message.toolFailed { return Theme.red }
        if !message.isComplete { return Theme.dim }
        return Theme.green
    }

    /// Argomenti compattati su una riga: {"command": "ls"} → command: "ls"
    private var compactArgs: String {
        var text = message.toolArgs
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        text = text.trimmingCharacters(in: CharacterSet(charactersIn: "{} "))
        if text.count > 90 {
            text = String(text.prefix(90)) + "…"
        }
        return text
    }

    private var resultLines: [String] {
        message.content.components(separatedBy: "\n")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("⏺")
                    .font(Theme.mono(12))
                    .foregroundStyle(dotColor)
                Text("\(message.toolName)(\(compactArgs))")
                    .font(Theme.mono(12, weight: .medium))
                    .foregroundStyle(Theme.secondary)
                    .lineLimit(1)
                    .textSelection(.enabled)
                if !message.isComplete {
                    ProgressView()
                        .controlSize(.mini)
                        .tint(Theme.accent)
                }
            }

            if !message.content.isEmpty {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() }
                } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("⎿")
                            .font(Theme.mono(12))
                            .foregroundStyle(Theme.dim)
                        if expanded {
                            Text(message.content)
                                .font(Theme.mono(11))
                                .foregroundStyle(resultColor)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        } else {
                            Text(previewText)
                                .font(Theme.mono(11))
                                .foregroundStyle(resultColor)
                                .lineLimit(1)
                            if resultLines.count > 1 {
                                Text("(+\(resultLines.count - 1) righe)")
                                    .font(Theme.mono(10))
                                    .foregroundStyle(Theme.dim)
                            }
                        }
                    }
                }
                .buttonStyle(.plain)
                .padding(.leading, 16)
            }
        }
        .padding(.top, 4)
    }

    private var previewText: String {
        resultLines.first ?? ""
    }

    private var resultColor: Color {
        if message.toolDenied { return Theme.orange }
        if message.toolFailed { return Theme.red }
        return Theme.dim
    }
}

// MARK: - Pannello di approvazione inline

struct ApprovalPanel: View {
    let request: ApprovalRequest
    let onDecision: (Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("⏺")
                    .foregroundStyle(Theme.orange)
                Text("l'agente vuole eseguire: \(request.toolName)")
                    .font(Theme.mono(12, weight: .semibold))
                    .foregroundStyle(Theme.text)
                Spacer()
            }

            ScrollView {
                Text(request.detail)
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 130)

            HStack {
                Text("consentire?")
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.dim)
                Spacer()
                Button("nega (esc)") { onDecision(false) }
                    .keyboardShortcut(.cancelAction)
                    .font(Theme.mono(11))
                Button("consenti (⌘⏎)") { onDecision(true) }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .font(Theme.mono(11))
            }
        }
        .padding(12)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Theme.orange.opacity(0.6), lineWidth: 1)
        )
        .padding(.horizontal, 14)
        .padding(.top, 8)
    }
}

// MARK: - Impostazioni

struct SettingsPane: View {
    @Bindable var engine: QwenEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Impostazioni")
                .font(Theme.mono(13, weight: .semibold))

            VStack(alignment: .leading, spacing: 4) {
                Text("temperatura: \(engine.temperature, specifier: "%.2f")")
                    .font(Theme.mono(12))
                Slider(value: $engine.temperature, in: 0...1.5)
                    .tint(Theme.accent)
            }

            if engine.activeModel?.supportsThinking ?? false {
                Toggle("modalità ragionamento (thinking)", isOn: $engine.enableThinking)
                    .toggleStyle(.switch)
                    .font(Theme.mono(12))
            }

            Divider()

            Toggle("modalità agente (tools)", isOn: $engine.agentMode)
                .toggleStyle(.switch)
                .font(Theme.mono(12))
            if engine.agentMode {
                if engine.workspaceURL == nil {
                    Text("scegli un Workspace dalla toolbar per attivarla")
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.orange)
                }
                HStack(spacing: 6) {
                    Text(engine.writeBoundaryURL.map { "scritture solo in: \($0.lastPathComponent)/" }
                        ?? "scritture in tutto il workspace")
                        .font(Theme.mono(11))
                        .foregroundStyle(engine.writeBoundaryURL != nil ? Theme.green : Theme.secondary)
                    Spacer()
                    Button("scegli…") { pickWriteBoundary() }
                        .font(Theme.mono(10))
                    if engine.writeBoundaryURL != nil {
                        Button("rimuovi") { engine.setWriteBoundary(nil) }
                            .font(Theme.mono(10))
                    }
                }
                Toggle("auto-approva scritture e comandi", isOn: $engine.autoApprove)
                    .toggleStyle(.switch)
                    .font(Theme.mono(12))
                if engine.autoApprove {
                    Text("⚠ il modello potrà modificare file ed eseguire comandi senza conferma")
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.orange)
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("limite memoria GPU: \(Int(engine.gpuMemoryLimitGB)) GB  (RAM: \(Int(QwenEngine.physicalRAMGB)) GB)")
                    .font(Theme.mono(12))
                Slider(
                    value: $engine.gpuMemoryLimitGB,
                    in: 4...max(8, QwenEngine.physicalRAMGB * 0.9), step: 1
                )
                .tint(Theme.accent)
                Text("tetto rigido: oltre questo limite la generazione si ferma con un errore, invece di bloccare il Mac")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("cache buffer GPU: \(Int(engine.gpuCacheLimitMB)) MB")
                    .font(Theme.mono(12))
                Slider(value: $engine.gpuCacheLimitMB, in: 16...1024, step: 16)
                    .tint(Theme.accent)
                Text("più alta = più veloce, più bassa = meno RAM occupata tra le risposte")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
            }

            Text("i limiti si applicano immediatamente")
                .font(Theme.mono(11))
                .foregroundStyle(Theme.dim)

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("prompt di sistema aggiuntivo")
                    .font(Theme.mono(12))
                TextEditor(text: $engine.systemPrompt)
                    .font(Theme.mono(12))
                    .frame(height: 80)
                    .scrollContentBackground(.hidden)
                    .background(Theme.inputBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Theme.border, lineWidth: 1)
                    )
            }

            Text("le modifiche valgono dalla prossima “nuova chat”")
                .font(Theme.mono(11))
                .foregroundStyle(Theme.dim)
        }
        .padding(16)
        .frame(width: 340)
    }
}


extension SettingsPane {
    fileprivate func pickWriteBoundary() {
        guard let workspace = engine.workspaceURL else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = workspace
        panel.message = "L'agente potrà scrivere SOLO in questa cartella (dentro il workspace)"
        panel.prompt = "Limita scritture qui"
        if panel.runModal() == .OK, let url = panel.url {
            // Il confine deve stare dentro il workspace, altrimenti nessuna
            // scrittura sarebbe mai permessa.
            let root = workspace.standardizedFileURL.resolvingSymlinksInPath().path
            let chosen = url.standardizedFileURL.resolvingSymlinksInPath().path
            if chosen == root || chosen.hasPrefix(root + "/") {
                engine.setWriteBoundary(url)
            }
        }
    }
}

// MARK: - Segmenti testo/codice della risposta

enum MessageSegment {
    case text(String)
    case code(language: String, body: String)
}

extension AssistantRow {
    /// Divide la risposta sui delimitatori ``` in segmenti testo e blocchi di codice.
    /// Un blocco non ancora chiuso (streaming in corso) è trattato come codice.
    static func segments(from content: String) -> [MessageSegment] {
        var result: [MessageSegment] = []
        var rest = Substring(content)
        while let fence = rest.range(of: "```") {
            let before = rest[..<fence.lowerBound]
            if !before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(.text(String(before).trimmingCharacters(in: .newlines)))
            }
            rest = rest[fence.upperBound...]
            let languageEnd = rest.firstIndex(of: "\n") ?? rest.endIndex
            let language = String(rest[..<languageEnd]).trimmingCharacters(in: .whitespaces)
            rest = languageEnd < rest.endIndex ? rest[rest.index(after: languageEnd)...] : Substring("")
            if let close = rest.range(of: "```") {
                result.append(.code(language: language, body: String(rest[..<close.lowerBound])))
                rest = rest[close.upperBound...]
            } else {
                result.append(.code(language: language, body: String(rest)))
                rest = Substring("")
            }
        }
        if !rest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.append(.text(String(rest).trimmingCharacters(in: .newlines)))
        }
        return result
    }
}

/// Blocco di codice con intestazione: linguaggio + copia + salva su file.
struct CodeBlockView: View {
    let language: String
    let body_: String
    @State private var copied = false

    init(language: String, body: String) {
        self.language = language
        self.body_ = body
    }

    private static let extensions: [String: String] = [
        "swift": "swift", "python": "py", "py": "py", "javascript": "js", "js": "js",
        "typescript": "ts", "ts": "ts", "html": "html", "css": "css", "json": "json",
        "bash": "sh", "sh": "sh", "zsh": "sh", "shell": "sh", "yaml": "yml", "yml": "yml",
        "markdown": "md", "md": "md", "sql": "sql", "c": "c", "cpp": "cpp", "java": "java",
        "kotlin": "kt", "rust": "rs", "go": "go", "ruby": "rb", "xml": "xml",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(language.isEmpty ? "codice" : language)
                    .font(Theme.mono(10, weight: .semibold))
                    .foregroundStyle(Theme.dim)
                Spacer()
                Button(copied ? "copiato ✓" : "copia") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(body_, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                }
                .buttonStyle(.plain)
                .font(Theme.mono(10))
                .foregroundStyle(copied ? Theme.green : Theme.secondary)
                Button("salva…") { save() }
                    .buttonStyle(.plain)
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Theme.panel)

            ScrollView(.horizontal) {
                Text(body_)
                    .font(Theme.mono(12))
                    .foregroundStyle(Theme.text)
                    .textSelection(.enabled)
                    .padding(10)
            }
        }
        .background(Theme.inputBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Theme.border.opacity(0.6), lineWidth: 1)
        )
        .padding(.vertical, 2)
    }

    private func save() {
        let panel = NSSavePanel()
        let ext = Self.extensions[language.lowercased()] ?? "txt"
        panel.nameFieldStringValue = "codice.\(ext)"
        panel.message = "Salva il blocco di codice"
        if panel.runModal() == .OK, let url = panel.url {
            try? body_.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

// MARK: - Pannello RAG

struct RAGPane: View {
    @Bindable var rag: RAGManager
    var engine: QwenEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("ricerca semantica (RAG)")
                .font(Theme.mono(13, weight: .semibold))

            if let workspace = engine.workspaceURL {
                statusRow

                HStack(spacing: 10) {
                    Button(rag.isReady ? "aggiorna indice" : "costruisci indice") {
                        rag.buildIndex(workspace: workspace)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .font(Theme.mono(11))
                    .disabled(isWorking)

                    if isWorking {
                        Button("annulla") { rag.cancelIndexing() }
                            .font(Theme.mono(11))
                    }
                }

                Text("indicizza testi, markdown, PDF e codice del workspace (embedding multilingue e5-small, ~450 MB al primo uso). L'indice vive in .localai/ dentro il workspace; il tool search_documents è attivo nelle nuove chat.")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("scegli prima un Workspace dalla toolbar")
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.orange)
            }
        }
        .padding(16)
        .frame(width: 340)
    }

    private var isWorking: Bool {
        switch rag.state {
        case .indexing, .loadingModel: true
        default: false
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        switch rag.state {
        case .idle:
            label("○", "nessun indice per questo workspace", Theme.dim)
        case .loadingModel(let fraction):
            VStack(alignment: .leading, spacing: 4) {
                label("✻", "scarico/carico l'embedder…", Theme.accent)
                ProgressView(value: max(fraction, 0.02)).tint(Theme.accent)
            }
        case .indexing(let done, let total):
            VStack(alignment: .leading, spacing: 4) {
                label("✻", "indicizzo… \(done)/\(total) frammenti", Theme.accent)
                ProgressView(value: Double(done), total: Double(max(total, 1))).tint(Theme.accent)
            }
        case .ready(let chunks):
            label("⏺", "indice pronto — \(chunks) frammenti", Theme.green)
        case .failed(let message):
            label("✗", message, Theme.red)
        }
    }

    private func label(_ dot: String, _ text: String, _ color: Color) -> some View {
        HStack(spacing: 8) {
            Text(dot).foregroundStyle(color)
            Text(text)
                .font(Theme.mono(11))
                .foregroundStyle(Theme.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}


// MARK: - Immagine generata

struct GeneratedImageView: View {
    let path: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 420, maxHeight: 420)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Theme.border.opacity(0.6), lineWidth: 1)
                    )
            } else {
                Text("⚠️ immagine non trovata: \(path)")
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.orange)
            }
            HStack(spacing: 12) {
                Button("salva…") { save() }
                    .buttonStyle(.plain)
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.secondary)
                Button("Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
                .buttonStyle(.plain)
                .font(Theme.mono(10))
                .foregroundStyle(Theme.secondary)
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
            }
        }
        .padding(.vertical, 4)
    }

    private func save() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
        panel.message = "Salva l'immagine generata"
        if panel.runModal() == .OK, let destination = panel.url {
            try? FileManager.default.copyItem(
                at: URL(fileURLWithPath: path), to: destination)
        }
    }
}
