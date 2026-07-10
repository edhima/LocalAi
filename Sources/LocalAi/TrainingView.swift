//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Finestra di fine-tuning: setup strumenti → dati → preparazione AI → LoRA → fusione.
struct TrainingView: View {
    @Bindable var manager: TrainingManager
    @Bindable var engine: QwenEngine
    @State private var convertSource = ""
    @State private var convertBits = 4

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    toolsSection
                    Divider()
                    dataSection
                    Divider()
                    trainingSection
                    Divider()
                    fuseSection
                    Divider()
                    exportSection
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minWidth: 380, idealWidth: 420)

            logConsole
                .frame(minWidth: 320)
        }
        .background(Theme.background)
        .navigationTitle("Training — LocalAi")
        .onAppear { manager.refreshEnvironment() }
    }

    // MARK: 1. Strumenti

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("1 · strumenti (mlx-lm)")
            HStack(spacing: 8) {
                Text("⏺")
                    .foregroundStyle(manager.toolsReady ? Theme.green : Theme.orange)
                Text(manager.usesEmbeddedRuntime
                    ? "Python + mlx-lm integrati nell'app"
                    : (manager.toolsReady ? "mlx-lm pronto (venv esterno)" : "mlx-lm non disponibile"))
                    .font(Theme.mono(12))
                    .foregroundStyle(Theme.secondary)
                Spacer()
                if !manager.usesEmbeddedRuntime {
                    Button(manager.toolsReady ? "aggiorna" : "installa") {
                        manager.setupTools()
                    }
                    .disabled(manager.isBusy)
                    .font(Theme.mono(11))
                }
            }
            Text(manager.usesEmbeddedRuntime
                ? "nessuna dipendenza esterna: disinstallare l'app rimuove tutto"
                : "runtime integrato assente (binario di sviluppo): fallback su venv")
                .font(Theme.mono(10))
                .foregroundStyle(Theme.dim)
        }
    }

    // MARK: 2. Dati

    private var dataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("2 · dati")
            HStack {
                Button("aggiungi file…") { pickFiles() }
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy)
                Text("txt · md · csv · json · pdf — i .jsonl già in formato chat entrano diretti")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.dim)
            }

            ForEach(manager.rawFiles, id: \.self) { file in
                HStack(spacing: 6) {
                    Text("·")
                    Text(file.lastPathComponent)
                    Spacer()
                    Button {
                        manager.rawFiles.removeAll { $0 == file }
                    } label: {
                        Image(systemName: "xmark").font(.system(size: 8))
                    }
                    .buttonStyle(.plain)
                }
                .font(Theme.mono(11))
                .foregroundStyle(Theme.secondary)
            }

            HStack(spacing: 10) {
                Button("prepara col modello caricato") {
                    manager.prepareData(engine: engine)
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accent)
                .font(Theme.mono(11))
                .disabled(manager.isBusy || manager.rawFiles.isEmpty || engine.phase != .ready)

                Stepper("esempi/blocco: \(manager.examplesPerChunk)",
                        value: $manager.examplesPerChunk, in: 1...8)
                    .font(Theme.mono(11))
            }
            if engine.phase != .ready {
                Text("⚠ carica prima un modello nella finestra principale (fa lui la preparazione)")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.orange)
            }
            if manager.totalChunks > 0, manager.isBusy {
                ProgressView(value: Double(manager.preparedChunks), total: Double(manager.totalChunks))
                    .tint(Theme.accent)
            }
            Text("dataset: \(manager.trainCount) train · \(manager.validCount) valid")
                .font(Theme.mono(11))
                .foregroundStyle(manager.trainCount > 0 ? Theme.green : Theme.dim)
        }
    }

    // MARK: 3. Training

    private var trainingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("3 · training LoRA")

            DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        TextField("org/modello-hf oppure /percorso/cartella", text: $convertSource)
                            .textFieldStyle(.plain)
                            .font(Theme.mono(11))
                            .padding(6)
                            .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 6))
                        Button("cartella…") { pickConvertFolder() }
                            .font(Theme.mono(10))
                    }
                    Picker("", selection: $convertBits) {
                        Text("4-bit (per QLoRA, consigliato)").tag(4)
                        Text("8-bit").tag(8)
                        Text("precisione originale").tag(0)
                    }
                    .pickerStyle(.radioGroup)
                    .labelsHidden()
                    .font(Theme.mono(11))
                    Button("converti in MLX") {
                        manager.convertModel(
                            source: convertSource,
                            quantizeBits: convertBits == 0 ? nil : convertBits)
                        convertSource = ""
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy || convertSource.trimmingCharacters(in: .whitespaces).isEmpty)
                    Text("serve un checkpoint HF completo (config + tokenizer + safetensors); l'output finisce in LocalAiTraining/converted/ e diventa il modello del training")
                        .font(Theme.mono(9))
                        .foregroundStyle(Theme.dim)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 6)
            } label: {
                Text("converti un modello HF in MLX (per training e chat)…")
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.secondary)
            }

            HStack(spacing: 6) {
                Text("modello")
                    .font(Theme.mono(11))
                    .foregroundStyle(Theme.dim)
                TextField("mlx-community/…", text: $manager.trainModelID)
                    .textFieldStyle(.plain)
                    .font(Theme.mono(11))
                    .padding(6)
                    .background(Theme.inputBackground, in: RoundedRectangle(cornerRadius: 6))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("iterazioni: \(Int(manager.iters))").font(Theme.mono(11))
                Slider(value: $manager.iters, in: 100...3000, step: 100).tint(Theme.accent)
                Text("batch: \(Int(manager.batchSize))").font(Theme.mono(11))
                Slider(value: $manager.batchSize, in: 1...8, step: 1).tint(Theme.accent)
                Text(String(format: "learning rate: %.0e", manager.learningRate)).font(Theme.mono(11))
                Slider(value: $manager.learningRate, in: 1e-6...1e-4).tint(Theme.accent)
            }

            HStack(spacing: 10) {
                Button("avvia training") { manager.startTraining() }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy || !manager.toolsReady || manager.trainCount == 0)
                if manager.isBusy {
                    Button("interrompi") { manager.cancel() }
                        .font(Theme.mono(11))
                }
            }

            if manager.currentIter > 0 {
                ProgressView(value: Double(manager.currentIter), total: manager.iters)
                    .tint(Theme.accent)
                HStack(spacing: 12) {
                    Text("iter \(manager.currentIter)/\(Int(manager.iters))")
                    if let loss = manager.lastTrainLoss {
                        Text(String(format: "train loss %.3f", loss))
                    }
                    if let loss = manager.lastValLoss {
                        Text(String(format: "val loss %.3f", loss)).foregroundStyle(Theme.green)
                    }
                }
                .font(Theme.mono(11))
                .foregroundStyle(Theme.secondary)
            }
        }
    }

    // MARK: 4. Fusione

    private var fuseSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("4 · fusione e uso")
            HStack(spacing: 10) {
                Button("fondi adapter nel modello") { manager.fuse() }
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy || !manager.hasAdapters)
                if let fused = manager.fusedModelURL {
                    Button("carica in chat") {
                        engine.loadLocalModel(directory: fused)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.green)
                    .font(Theme.mono(11))
                    .disabled(engine.phase.isBusy)
                }
            }
            if let fused = manager.fusedModelURL {
                Text("modello fuso: \(fused.path)")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.green)
                    .lineLimit(1)
                    .truncationMode(.middle)
            } else if manager.hasAdapters {
                Text("adapters pronti — puoi fonderli in un modello autonomo")
                    .font(Theme.mono(10))
                    .foregroundStyle(Theme.secondary)
            }
            Text("progetto: \(manager.projectURL.path)")
                .font(Theme.mono(10))
                .foregroundStyle(Theme.dim)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    // MARK: 5. Export produzione

    private var exportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("5 · esporta per sistemi live")
            HStack(spacing: 10) {
                Button("fp16 → vLLM/transformers") { manager.exportFP16() }
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy || !manager.hasAdapters)
                Button("GGUF → Ollama/llama.cpp") { manager.exportGGUF() }
                    .font(Theme.mono(11))
                    .disabled(manager.isBusy || !manager.hasAdapters)
            }

            if let fp16 = manager.fp16ExportURL {
                exportRow("fp16: \(fp16.lastPathComponent)/ — safetensors HF standard", fp16)
            }
            if let gguf = manager.ggufExportURL {
                exportRow("gguf: \(gguf.lastPathComponent)", gguf)
            }

            Text("fp16 = pesi dequantizzati standard, li carichi con vLLM/TGI/transformers. GGUF diretto solo per architetture supportate da mlx-lm; per le altre: fp16 + convert_hf_to_gguf.py di llama.cpp. Per il modello di produzione definitivo valuta il retraining sulla base fp16.")
                .font(Theme.mono(10))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func exportRow(_ label: String, _ url: URL) -> some View {
        HStack(spacing: 8) {
            Text("⏺").foregroundStyle(Theme.green)
            Text(label)
                .font(Theme.mono(10))
                .foregroundStyle(Theme.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
            .font(Theme.mono(10))
        }
    }

    // MARK: Console log

    private var logConsole: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(manager.isBusy ? "✻ \(manager.busyLabel)" : "log")
                    .font(Theme.mono(11, weight: .semibold))
                    .foregroundStyle(manager.isBusy ? Theme.accent : Theme.dim)
                Spacer()
            }
            .padding(10)

            ScrollViewReader { proxy in
                ScrollView {
                    Text(manager.log.isEmpty ? "(il log di preparazione e training appare qui)" : manager.log)
                        .font(Theme.mono(10))
                        .foregroundStyle(Theme.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .id("logEnd")
                }
                .onChange(of: manager.log) {
                    proxy.scrollTo("logEnd", anchor: .bottom)
                }
            }
            .background(Theme.inputBackground)
        }
        .background(Theme.panel)
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(Theme.mono(12, weight: .semibold))
            .foregroundStyle(Theme.text)
    }

    private func pickConvertFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli la cartella del modello in formato Hugging Face"
        if panel.runModal() == .OK, let url = panel.url {
            convertSource = url.path
        }
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        var types: [UTType] = [.plainText, .utf8PlainText, .json, .pdf, .commaSeparatedText]
        if let markdown = UTType(filenameExtension: "md") { types.append(markdown) }
        if let jsonl = UTType(filenameExtension: "jsonl") { types.append(jsonl) }
        panel.allowedContentTypes = types
        panel.message = "Scegli i file di partenza per il dataset di training"
        if panel.runModal() == .OK {
            manager.addRawFiles(panel.urls)
        }
    }
}