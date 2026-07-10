//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import AppKit
import SwiftUI

@main
struct LocalAiApp: App {
    @State private var engine = QwenEngine()
    @State private var store = ModelStore()
    @State private var training = TrainingManager()
    @State private var rag = RAGManager()
    @State private var imageGen = ImageGenManager()
    @State private var imageTrainer = ImageTrainingManager()

    init() {
        // Lanciata come binario nudo (senza bundle .app) serve la activation
        // policy per far comparire finestra, menu e icona nel Dock.
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    var body: some Scene {
        WindowGroup("LocalAi") {
            ContentView(engine: engine, store: store, rag: rag, imageGen: imageGen)
                .frame(minWidth: 960, minHeight: 600)
                .preferredColorScheme(.dark)
                .tint(Theme.accent)
        }

        Window("Training", id: "training") {
            TrainingView(manager: training, engine: engine, imageTrainer: imageTrainer, imageGen: imageGen)
                .frame(minWidth: 780, minHeight: 560)
                .preferredColorScheme(.dark)
                .tint(Theme.accent)
        }
    }
}