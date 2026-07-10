//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import Foundation

/// Un modello Qwen disponibile nel catalogo, in formato MLX (mlx-community su Hugging Face).
public struct CatalogModel: Identifiable, Hashable, Sendable, Codable {
    /// ID del repository Hugging Face, es. "mlx-community/Qwen3-4B-4bit"
    public let id: String
    public let displayName: String
    /// Dimensione approssimativa su disco, in GB.
    public let sizeGB: Double
    /// Modelli Qwen3 con supporto al "thinking" (blocchi <think>...</think>).
    public let supportsThinking: Bool
    public let note: String

    public init(id: String, displayName: String, sizeGB: Double, supportsThinking: Bool, note: String) {
        self.id = id
        self.displayName = displayName
        self.sizeGB = sizeGB
        self.supportsThinking = supportsThinking
        self.note = note
    }

    public var shortName: String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    public var sizeText: String {
        if sizeGB == 0 { return "" }
        return sizeGB < 1
            ? String(format: "~%.0f MB", sizeGB * 1024)
            : String(format: "~%.1f GB", sizeGB)
    }

    /// Vero se punta a una cartella locale invece che a un repo Hugging Face.
    public var isLocalDirectory: Bool { id.hasPrefix("local:") }
}

public enum ModelCatalog {
    /// Modelli curati per una macchina Apple Silicon con molta RAM (quantizzazione 4 bit).
    public static let models: [CatalogModel] = [
        CatalogModel(
            id: "mlx-community/Qwen3-0.6B-4bit",
            displayName: "Qwen3 0.6B",
            sizeGB: 0.35,
            supportsThinking: true,
            note: "Minuscolo, ideale per un primo test"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-1.7B-4bit",
            displayName: "Qwen3 1.7B",
            sizeGB: 1.0,
            supportsThinking: true,
            note: "Velocissimo, qualità discreta"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-4B-4bit",
            displayName: "Qwen3 4B",
            sizeGB: 2.3,
            supportsThinking: true,
            note: "Il più piccolo affidabile come agente"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen3 8B",
            sizeGB: 4.7,
            supportsThinking: true,
            note: "Uso quotidiano di qualità"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-14B-4bit",
            displayName: "Qwen3 14B",
            sizeGB: 8.4,
            supportsThinking: true,
            note: "Alta qualità, ancora reattivo"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-30B-A3B-4bit",
            displayName: "Qwen3 30B (MoE A3B)",
            sizeGB: 17.2,
            supportsThinking: true,
            note: "MoE: qualità da 30B, velocità da 3B"
        ),
        CatalogModel(
            id: "mlx-community/Qwen3-32B-4bit",
            displayName: "Qwen3 32B",
            sizeGB: 18.4,
            supportsThinking: true,
            note: "Il più capace, più lento"
        ),
        CatalogModel(
            id: "mlx-community/GLM-4.7-Flash-4bit",
            displayName: "GLM-4.7 Flash (MoE)",
            sizeGB: 16.9,
            supportsThinking: true,
            note: "Il GLM più recente che gira in locale"
        ),
        CatalogModel(
            id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            displayName: "Qwen2.5 Coder 7B",
            sizeGB: 4.3,
            supportsThinking: false,
            note: "Specializzato in codice"
        ),
    ]
}