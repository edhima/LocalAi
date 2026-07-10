//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import QwenLocalCore
import AppKit
import SwiftUI

struct ContentView: View {
    @Bindable var engine: QwenEngine
    var store: ModelStore
    var rag: RAGManager
    var imageGen: ImageGenManager
    var training: TrainingManager

    var body: some View {
        NavigationSplitView {
            SidebarView(engine: engine, store: store, imageGen: imageGen, training: training)
                .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 400)
        } detail: {
            ChatView(engine: engine, rag: rag, imageGen: imageGen)
        }
        .background(Theme.background)
    }
}

// MARK: - Sidebar: catalogo e gestione modelli

struct SidebarView: View {
    @Bindable var engine: QwenEngine
    var store: ModelStore
    @Bindable var imageGen: ImageGenManager
    @Bindable var training: TrainingManager
    @Environment(\.openWindow) private var openWindow
    @State private var modelToDelete: CatalogModel?
    @State private var newModelID = ""
    @State private var mountError: String?
    @State private var inspectionReport: String?

    private func addByID() {
        store.addCustom(id: newModelID)
        newModelID = ""
    }

    private func addLocalFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli una cartella-modello MLX (config.json + safetensors)"
        if panel.runModal() == .OK, let url = panel.url {
            mountError = store.addCustomLocal(url: url)
        }
    }

    private func inspectSafetensors() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli un file .safetensors da identificare"
        if panel.runModal() == .OK, let url = panel.url {
            inspectionReport = SafetensorsInspector.inspect(url: url)
        }
    }

    var body: some View {
        List {
            Section {
                ForEach(ModelCatalog.models) { model in
                    ModelRow(
                        model: model,
                        engine: engine,
                        store: store,
                        onDelete: { modelToDelete = model }
                    )
                }
            } header: {
                Text("catalogo modelli")
                    .font(Theme.mono(11, weight: .semibold))
                    .foregroundStyle(Theme.dim)
            }

            Section {
                ForEach(store.customModels) { model in
                    ModelRow(
                        model: model,
                        engine: engine,
                        store: store,
                        onDelete: { modelToDelete = model }
                    )
                    .contextMenu {
                        Button("Rimuovi dalla lista") {
                            if engine.activeModel?.id == model.id {
                                engine.unload()
                            }
                            store.removeCustom(id: model.id)
                        }
                    }
                }

                HStack(spacing: 6) {
                    TextField("org/nome-modello-mlx", text: $newModelID)
                        .textFieldStyle(.plain)
                        .font(Theme.mono(11))
                        .onSubmit(addByID)
                    Button(action: addByID) {
                        Image(systemName: "plus.circle")
                    }
                    .buttonStyle(.borderless)
                    .disabled(!newModelID.contains("/"))
                    .help("Aggancia un repo Hugging Face in formato MLX")
                }
                Button {
                    addLocalFolder()
                } label: {
                    Label("cartella modello locale…", systemImage: "folder.badge.plus")
                        .font(Theme.mono(11))
                }
                .buttonStyle(.borderless)
                .help("Aggancia una cartella-modello MLX (es. un tuo fine-tuning fuso)")
                Button {
                    inspectSafetensors()
                } label: {
                    Label("ispeziona safetensors…", systemImage: "doc.text.magnifyingglass")
                        .font(Theme.mono(11))
                }
                .buttonStyle(.borderless)
                .help("Identifica un file .safetensors: checkpoint o LoRA, architettura, parametri")
                Button {
                    training.windowMode = 0
                    openWindow(id: "training")
                } label: {
                    Label("training LoRA (testo)…", systemImage: "graduationcap")
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.accent)
                }
                .buttonStyle(.borderless)
                .help("Fine-tuning LoRA di un modello linguistico sui tuoi dati")
            } header: {
                Text("altri modelli")
                    .font(Theme.mono(11, weight: .semibold))
                    .foregroundStyle(Theme.dim)
            }

            Section {
                ImageGenRow(imageGen: imageGen, engine: engine)
                Button {
                    training.windowMode = 1
                    openWindow(id: "training")
                } label: {
                    Label("training LoRA (immagini)…", systemImage: "graduationcap")
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.accent)
                }
                .buttonStyle(.borderless)
                .help("Addestra un LoRA SDXL con le tue immagini (stile DreamBooth)")
            } header: {
                Text("modello immagine (SD/SDXL)")
                    .font(Theme.mono(11, weight: .semibold))
                    .foregroundStyle(Theme.dim)
            }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.panel)
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 3) {
                Divider()
                Text("cache: \(ModelStore.hubCacheURL.path.replacingOccurrences(of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~"))")
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("spazio usato: \(store.totalSizeText)")
            }
            .font(Theme.mono(10))
            .foregroundStyle(Theme.dim)
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
        }
        .navigationTitle("LocalAi")
        .confirmationDialog(
            "Eliminare \(modelToDelete?.displayName ?? "") dal disco?",
            isPresented: Binding(
                get: { modelToDelete != nil },
                set: { if !$0 { modelToDelete = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button("Elimina", role: .destructive) {
                if let model = modelToDelete {
                    if engine.activeModel?.id == model.id {
                        engine.unload()
                    }
                    store.delete(model.id)
                }
                modelToDelete = nil
            }
            Button("Annulla", role: .cancel) { modelToDelete = nil }
        } message: {
            Text("I file verranno rimossi dalla cache di Hugging Face. Potrai riscaricarli in qualsiasi momento.")
        }
        .alert(
            "Cartella non montabile",
            isPresented: Binding(
                get: { mountError != nil },
                set: { if !$0 { mountError = nil } }
            )
        ) {
            Button("OK") { mountError = nil }
        } message: {
            Text(mountError ?? "")
        }
        .sheet(isPresented: Binding(
            get: { inspectionReport != nil },
            set: { if !$0 { inspectionReport = nil } }
        )) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Referto safetensors", systemImage: "doc.text.magnifyingglass")
                    .font(Theme.mono(13, weight: .semibold))
                ScrollView {
                    Text(inspectionReport ?? "")
                        .font(Theme.mono(11))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 260)
                .padding(10)
                .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 8))
                HStack {
                    Spacer()
                    Button("Chiudi") { inspectionReport = nil }
                        .keyboardShortcut(.defaultAction)
                }
            }
            .padding(16)
            .frame(width: 520)
        }
    }
}

struct ModelRow: View {
    let model: CatalogModel
    @Bindable var engine: QwenEngine
    var store: ModelStore
    var onDelete: () -> Void

    private var isActive: Bool { engine.activeModel?.id == model.id }
    private var isDownloaded: Bool {
        model.isLocalDirectory
            ? FileManager.default.fileExists(
                atPath: String(model.id.dropFirst("local:".count)) + "/config.json")
            : store.isDownloaded(model.id)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                if isActive {
                    Text("⏺")
                        .font(Theme.mono(11))
                        .foregroundStyle(statusColor)
                }
                Text(model.displayName)
                    .font(Theme.mono(12, weight: isActive ? .semibold : .regular))
                    .foregroundStyle(isActive ? Theme.text : Theme.secondary)
                Spacer()
                actions
            }
            Text(model.note)
                .font(Theme.mono(10))
                .foregroundStyle(Theme.dim)
            HStack(spacing: 6) {
                if model.isLocalDirectory {
                    Text("cartella locale")
                } else {
                    Text(store.diskSizeText(model.id) ?? model.sizeText)
                    if isDownloaded {
                        Text("· sul disco")
                    }
                }
            }
            .font(Theme.mono(10))
            .foregroundStyle(Theme.dim)

            if isActive, case .downloading(let fraction) = engine.phase {
                ProgressView(value: fraction)
                    .progressViewStyle(.linear)
                    .tint(Theme.accent)
                Text("download… \(Int(fraction * 100))%")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.secondary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

    private var statusColor: Color {
        switch engine.phase {
        case .ready: Theme.green
        case .downloading, .loadingWeights: Theme.orange
        default: Theme.dim
        }
    }

    @ViewBuilder
    private var actions: some View {
        if isActive, engine.phase == .ready || engine.phase.isBusy {
            Button {
                engine.unload()
            } label: {
                Image(systemName: engine.phase.isBusy ? "xmark.circle" : "eject")
            }
            .buttonStyle(.borderless)
            .help(engine.phase.isBusy ? "Annulla" : "Scarica dalla memoria")
        } else {
            Button {
                engine.load(model) { [weak store] in store?.refresh() }
            } label: {
                Image(systemName: isDownloaded ? "play.circle" : "arrow.down.circle")
            }
            .buttonStyle(.borderless)
            .disabled(engine.phase.isBusy)
            .help(isDownloaded ? "Carica in memoria" : "Scarica (\(model.sizeText)) e carica")

            if isDownloaded, !model.isLocalDirectory {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("Elimina dal disco")
            }
        }
    }
}


// MARK: - Modello immagine (Stable Diffusion / SDXL)

struct ImageGenRow: View {
    @Bindable var imageGen: ImageGenManager
    @Bindable var engine: QwenEngine
    @State private var showParams = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            switch imageGen.state {
            case .toolsMissing:
                Text("stack immagini non installato (torch+diffusers, ~2,5 GB)")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
                Button("installa stack immagini") { imageGen.installTools() }
                    .font(Theme.mono(11))
            case .installingTools:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("installo… (vedi log sotto)")
                        .font(Theme.mono(10))
                        .foregroundStyle(Theme.secondary)
                }
            case .idle:
                Button {
                    pickCheckpoint()
                } label: {
                    Label("monta checkpoint…", systemImage: "photo.badge.plus")
                        .font(Theme.mono(11))
                }
                .buttonStyle(.borderless)
                if engine.activeModel != nil {
                    Text("⚠ montando il checkpoint verrà liberato il modello chat (\(engine.activeModel?.shortName ?? "")) per la memoria")
                        .font(Theme.mono(9))
                        .foregroundStyle(Theme.orange)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("un .safetensors SD/SDXL (es. da ComfyUI); genera dalla chat con /img")
                        .font(Theme.mono(10))
                        .foregroundStyle(Theme.dim)
                }
            case .loadingModel:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("carico la pipeline…")
                        .font(Theme.mono(10))
                        .foregroundStyle(Theme.secondary)
                }
            case .ready(let model):
                HStack(spacing: 6) {
                    Text("⏺").foregroundStyle(Theme.green)
                    Text(model)
                        .font(Theme.mono(11))
                        .foregroundStyle(Theme.text)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button {
                        showParams.toggle()
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .buttonStyle(.borderless)
                    .help("Parametri di generazione: passi, dimensioni, guidance, seed…")
                    .popover(isPresented: $showParams, arrowEdge: .trailing) {
                        ImageGenParamsPane(imageGen: imageGen)
                    }
                    Button {
                        imageGen.unmount()
                    } label: {
                        Image(systemName: "eject")
                    }
                    .buttonStyle(.borderless)
                }
                // Riga LoRA sempre visibile: è "l'add-on sopra il checkpoint",
                // non un secondo modello da montare a parte.
                if let lora = imageGen.loraURL {
                    HStack(spacing: 6) {
                        Text("+ LoRA").foregroundStyle(Theme.accent)
                        Text(lora.deletingLastPathComponent().lastPathComponent)
                            .foregroundStyle(Theme.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("togli") { imageGen.remount(lora: nil) }
                    }
                    .font(Theme.mono(10))
                } else {
                    Button {
                        pickLora()
                    } label: {
                        Label("aggiungi un LoRA sopra…", systemImage: "plus.rectangle.on.rectangle")
                            .font(Theme.mono(10))
                            .foregroundStyle(Theme.accent)
                    }
                    .buttonStyle(.borderless)
                }
                Text("pronto — scrivi /img <descrizione> in chat · \(imageGen.width)×\(imageGen.height) · \(Int(imageGen.steps)) passi")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.green)
            case .generating(let step, let total):
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(total > 0 ? "genero… \(step)/\(total)" : "genero…")
                        .font(Theme.mono(10))
                        .foregroundStyle(Theme.accent)
                }
            case .failed(let message):
                Text("✗ \(message)")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.red)
                    .fixedSize(horizontal: false, vertical: true)
                // Dettaglio reale (traceback del worker) per capire la causa
                if !imageGen.log.isEmpty {
                    ScrollView {
                        Text(String(imageGen.log.suffix(600)))
                            .font(Theme.mono(9))
                            .foregroundStyle(Theme.dim)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 90)
                }
                HStack(spacing: 8) {
                    Button {
                        pickCheckpoint()
                    } label: {
                        Label("scegli un altro file…", systemImage: "photo.badge.plus")
                            .font(Theme.mono(10))
                            .foregroundStyle(Theme.accent)
                    }
                    .buttonStyle(.borderless)
                    Button("riprova") { imageGen.refreshEnvironment() }
                        .font(Theme.mono(10))
                    if imageGen.loraURL != nil {
                        Button("senza LoRA") { imageGen.remount(lora: nil) }
                            .font(Theme.mono(10))
                    }
                }
            }

            if !imageGen.log.isEmpty, case .installingTools = imageGen.state {
                Text(String(imageGen.log.suffix(300)))
                    .font(Theme.mono(9))
                    .foregroundStyle(Theme.dim)
                    .lineLimit(4)
            }
        }
        .padding(.vertical, 4)
    }

    private func pickCheckpoint() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli un checkpoint Stable Diffusion / SDXL (.safetensors single-file)"
        if panel.runModal() == .OK, let url = panel.url {
            freeChatModelIfLoaded()
            imageGen.mount(checkpoint: url)
        }
    }

    /// Chat e immagini competono per la stessa memoria GPU su Apple Silicon:
    /// libera il modello chat prima di caricare la pipeline immagini.
    private func freeChatModelIfLoaded() {
        if engine.activeModel != nil {
            engine.unload()
        }
    }

    private func pickLora() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli un LoRA (.safetensors) da applicare sul checkpoint montato"
        if let docs = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            panel.directoryURL = docs.appendingPathComponent("LocalAiTraining/loras")
        }
        if panel.runModal() == .OK, let url = panel.url {
            imageGen.remount(lora: url)
        }
    }
}

/// Parametri di generazione del modello immagine montato.
struct ImageGenParamsPane: View {
    @Bindable var imageGen: ImageGenManager

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("parametri generazione (\(imageGen.mountedArch == "sd" ? "SD" : "SDXL"))")
                .font(Theme.mono(13, weight: .semibold))

            VStack(alignment: .leading, spacing: 4) {
                Text("passi: \(Int(imageGen.steps))  ·  guidance: \(imageGen.guidance, specifier: "%.1f")")
                    .font(Theme.mono(11))
                Slider(value: $imageGen.steps, in: 4...60, step: 1).tint(Theme.accent)
                Slider(value: $imageGen.guidance, in: 1...15, step: 0.5).tint(Theme.accent)
                Text("più passi = più dettaglio ma più lento · guidance alta = più fedele al prompt")
                    .font(Theme.mono(9))
                    .foregroundStyle(Theme.dim)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("dimensioni: \(imageGen.width) × \(imageGen.height)")
                    .font(Theme.mono(11))
                Picker("", selection: Binding(
                    get: { "\(imageGen.width)x\(imageGen.height)" },
                    set: { value in
                        if let preset = imageGen.sizePresets.first(where: { "\($0.width)x\($0.height)" == value }) {
                            imageGen.width = preset.width
                            imageGen.height = preset.height
                        }
                    }
                )) {
                    ForEach(imageGen.sizePresets, id: \.label) { preset in
                        Text(preset.label)
                            .font(Theme.mono(11))
                            .tag("\(preset.width)x\(preset.height)")
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
                .font(Theme.mono(11))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("negative prompt (cosa evitare)")
                    .font(Theme.mono(11))
                TextField("es. blurry, low quality, deformed…", text: $imageGen.negativePrompt)
                    .textFieldStyle(.plain)
                    .font(Theme.mono(11))
                    .padding(6)
                    .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 6))
            }

            HStack(spacing: 8) {
                Text("seed")
                    .font(Theme.mono(11))
                TextField("vuoto = casuale", text: $imageGen.seedText)
                    .textFieldStyle(.plain)
                    .font(Theme.mono(11))
                    .padding(6)
                    .frame(width: 120)
                    .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 6))
                if let last = imageGen.lastSeed {
                    Button("riusa ultimo (\(last))") {
                        imageGen.seedText = String(last)
                    }
                    .font(Theme.mono(10))
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("LoRA (add-on sul checkpoint)")
                    .font(Theme.mono(11, weight: .semibold))
                if let lora = imageGen.loraURL {
                    HStack(spacing: 6) {
                        Text("⏺").foregroundStyle(Theme.green)
                        Text(lora.deletingLastPathComponent().lastPathComponent)
                            .font(Theme.mono(10))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("togli") { imageGen.remount(lora: nil) }
                            .font(Theme.mono(10))
                    }
                    Text("scala: \(imageGen.loraScale, specifier: "%.2f")")
                        .font(Theme.mono(10))
                    Slider(value: $imageGen.loraScale, in: 0...1.5, step: 0.05)
                        .tint(Theme.accent)
                    Text("cambia la scala e rimonta per applicarla")
                        .font(Theme.mono(9))
                        .foregroundStyle(Theme.dim)
                    Button("rimonta con scala \(imageGen.loraScale, specifier: "%.2f")") {
                        imageGen.remount(lora: lora)
                    }
                    .font(Theme.mono(10))
                } else {
                    Button("applica un LoRA…") { pickLora() }
                        .font(Theme.mono(11))
                    Text("un pytorch_lora_weights.safetensors da LocalAiTraining/loras/ o ComfyUI")
                        .font(Theme.mono(9))
                        .foregroundStyle(Theme.dim)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Text("i parametri valgono da subito, per la prossima /img")
                .font(Theme.mono(10))
                .foregroundStyle(Theme.dim)
        }
        .padding(16)
        .frame(width: 340)
    }

    private func pickLora() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli un file LoRA (.safetensors) da applicare sul checkpoint montato"
        if let loras = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            panel.directoryURL = loras.appendingPathComponent("LocalAiTraining/loras")
        }
        if panel.runModal() == .OK, let url = panel.url {
            imageGen.remount(lora: url)
        }
    }
}
