//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import Foundation
import Observation
import QwenLocalCore

/// Gestisce i modelli scaricati nella cache Hugging Face su disco
/// (la stessa cache usata da Python e dalla CLI `hf`).
@MainActor
@Observable
final class ModelStore {
    private(set) var downloadedIDs: Set<String> = []
    private(set) var diskSizes: [String: Int64] = [:]

    /// Directory della cache hub, con la stessa logica di risoluzione di HubClient:
    /// HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub
    static var hubCacheURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: hubCache)
        }
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    // MARK: - Modelli personalizzati (agganciati dall'utente)

    private static let customKey = "customModels"
    private(set) var customModels: [CatalogModel] = []

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.customKey),
            let saved = try? JSONDecoder().decode([CatalogModel].self, from: data) {
            customModels = saved
        }
        refresh()
    }

    private func saveCustom() {
        if let data = try? JSONEncoder().encode(customModels) {
            UserDefaults.standard.set(data, forKey: Self.customKey)
        }
    }

    /// Aggancia un repo Hugging Face (es. "mlx-community/Llama-3.2-3B-Instruct-4bit").
    func addCustom(id: String) {
        let clean = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.contains("/"), !clean.hasPrefix("local:"),
            !customModels.contains(where: { $0.id == clean }),
            !ModelCatalog.models.contains(where: { $0.id == clean })
        else { return }
        customModels.append(CatalogModel(
            id: clean,
            displayName: clean.split(separator: "/").last.map(String.init) ?? clean,
            sizeGB: 0,
            supportsThinking: clean.localizedCaseInsensitiveContains("qwen3"),
            note: "modello agganciato (formato MLX)"))
        saveCustom()
        refresh()
    }

    /// Verifica che una cartella sia un modello linguistico MLX/HF caricabile.
    /// Restituisce nil se valida, altrimenti il motivo del rifiuto.
    static func validateModelDirectory(_ url: URL) -> String? {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.appendingPathComponent("model_index.json").path) {
            return "È una pipeline diffusers (generazione immagini/video), non un modello linguistico: usala con ComfyUI o diffusers."
        }
        let configURL = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return "Manca config.json: non è una cartella-modello in formato Hugging Face/MLX. Un singolo file .safetensors non basta (servono config e tokenizer) — usa “ispeziona safetensors…” per capire cos'è."
        }
        if config["_class_name"] != nil {
            return "Il config.json è di un modello diffusion (immagini/video), non di un modello linguistico."
        }
        guard config["model_type"] is String else {
            return "config.json senza model_type: architettura non riconoscibile come LLM."
        }
        let hasWeights = (try? fm.contentsOfDirectory(atPath: url.path))?
            .contains { $0.hasSuffix(".safetensors") } ?? false
        if !hasWeights {
            return "Nessun file .safetensors nella cartella: mancano i pesi del modello."
        }
        return nil
    }

    /// Aggancia una cartella-modello MLX locale (es. l'output di un fine-tuning).
    /// Restituisce un messaggio d'errore se la cartella non è un LLM caricabile.
    @discardableResult
    func addCustomLocal(url: URL) -> String? {
        if let problem = Self.validateModelDirectory(url) {
            return problem
        }
        let id = "local:" + url.standardizedFileURL.path
        guard !customModels.contains(where: { $0.id == id }) else { return nil }
        customModels.append(CatalogModel(
            id: id,
            displayName: url.lastPathComponent,
            sizeGB: 0,
            supportsThinking: true,
            note: "cartella locale: \(url.deletingLastPathComponent().lastPathComponent)/"))
        saveCustom()
        return nil
    }

    func removeCustom(id: String) {
        customModels.removeAll { $0.id == id }
        saveCustom()
    }

    func isDownloaded(_ id: String) -> Bool {
        downloadedIDs.contains(id)
    }

    func diskSizeText(_ id: String) -> String? {
        guard let bytes = diskSizes[id], bytes > 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var totalSizeText: String {
        let total = diskSizes.values.reduce(0, +)
        return ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
    }

    /// Riesamina la cache su disco. La struttura è quella standard HF:
    /// `models--org--nome/{blobs,snapshots,refs}`.
    func refresh() {
        let fm = FileManager.default
        var found: Set<String> = []
        var sizes: [String: Int64] = [:]

        let contents = (try? fm.contentsOfDirectory(
            at: Self.hubCacheURL, includingPropertiesForKeys: nil)) ?? []
        for dir in contents where dir.lastPathComponent.hasPrefix("models--") {
            let repoID = dir.lastPathComponent
                .replacingOccurrences(of: "models--", with: "")
                .replacingOccurrences(of: "--", with: "/")
            // Considera "scaricato" solo un repo con almeno uno snapshot non vuoto.
            let snapshots = dir.appendingPathComponent("snapshots")
            guard let snaps = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil),
                !snaps.isEmpty
            else { continue }
            found.insert(repoID)
            sizes[repoID] = Self.directorySize(dir)
        }
        downloadedIDs = found
        diskSizes = sizes
    }

    /// Elimina dal disco tutti i file di un modello.
    func delete(_ id: String) {
        let dirName = "models--" + id.replacingOccurrences(of: "/", with: "--")
        let dir = Self.hubCacheURL.appendingPathComponent(dirName)
        try? FileManager.default.removeItem(at: dir)
        refresh()
    }

    /// Somma dei byte dei file regolari (i pesi vivono in `blobs/`, gli snapshot sono symlink).
    private static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileAllocatedSizeKey, .isSymbolicLinkKey],
            options: []
        ) else { return 0 }

        var total: Int64 = 0
        for case let file as URL in enumerator {
            guard let values = try? file.resourceValues(
                forKeys: [.isRegularFileKey, .fileAllocatedSizeKey, .isSymbolicLinkKey]),
                values.isRegularFile == true, values.isSymbolicLink != true
            else { continue }
            total += Int64(values.fileAllocatedSize ?? 0)
        }
        return total
    }
}