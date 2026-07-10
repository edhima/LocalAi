import Foundation

/// Legge l'intestazione JSON di un file .safetensors (senza caricare i pesi)
/// e produce un referto leggibile: cos'è, per quale architettura, quanto pesa.
enum SafetensorsInspector {

    static func inspect(url: URL) -> String {
        let sizeBytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        let sizeMB = Double(sizeBytes ?? 0) / 1_048_576

        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return "✗ file illeggibile"
        }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else {
            return "✗ file troppo corto: non è un safetensors"
        }
        let headerLength = lengthData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard headerLength > 0, headerLength < 200_000_000,
            let headerData = try? handle.read(upToCount: Int(headerLength)),
            let raw = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            return "✗ intestazione non valida: non è un safetensors (o è corrotto)"
        }

        var tensors = raw
        let metadata = tensors.removeValue(forKey: "__metadata__") as? [String: Any] ?? [:]

        var parameters: Int64 = 0
        var dtypes = Set<String>()
        for value in tensors.values {
            guard let tensor = value as? [String: Any] else { continue }
            if let dtype = tensor["dtype"] as? String { dtypes.insert(dtype) }
            if let shape = tensor["shape"] as? [Int] {
                parameters += Int64(shape.reduce(1, *))
            }
        }

        let keys = Array(tensors.keys)
        let loraCount = keys.filter { $0.lowercased().contains("lora") }.count
        let kind: String
        if loraCount > 0, loraCount >= keys.count / 2 {
            kind = "LoRA (adattatore, si applica sopra un modello base)"
        } else if loraCount > 0 {
            kind = "checkpoint con tensori LoRA misti"
        } else {
            kind = "checkpoint completo"
        }

        let architecture = Self.architecture(from: keys)

        var report = """
        file        \(url.lastPathComponent)
        dimensione  \(String(format: "%.0f MB", sizeMB))
        tipo        \(kind)
        architettura \(architecture.name)
        tensori     \(keys.count) · dtype \(dtypes.sorted().joined(separator: ", "))
        parametri   ~\(Self.formatParams(parameters))
        utilizzabile in LocalAi: \(architecture.isLLM ? "SÌ, se accompagnato da config.json e tokenizer nella stessa cartella" : "NO — non è un modello linguistico")
        """
        if !architecture.isLLM {
            report += "\ndestinazione consigliata: \(architecture.destination)"
        }
        if !metadata.isEmpty {
            let interesting = metadata.keys.sorted().prefix(4)
                .map { "\($0)=\(String(describing: metadata[$0] ?? "").prefix(40))" }
                .joined(separator: " · ")
            report += "\nmetadata    \(interesting)"
        }
        return report
    }

    /// Riconosce la famiglia di architettura dai nomi dei tensori.
    private static func architecture(from keys: [String]) -> (name: String, isLLM: Bool, destination: String) {
        let sample = keys.prefix(400).joined(separator: "\n")
        if sample.contains("conditioner.embedders") || sample.contains("first_stage_model") {
            return ("Stable Diffusion XL (generazione immagini)", false, "ComfyUI → models/checkpoints/")
        }
        if sample.contains("model.diffusion_model") {
            return ("Stable Diffusion (generazione immagini)", false, "ComfyUI → models/checkpoints/")
        }
        if sample.contains("double_blocks") || sample.contains("single_blocks") {
            return ("Flux (generazione immagini)", false, "ComfyUI → models/diffusion_models/")
        }
        if sample.contains("pipe.dit") || sample.contains("dit.blocks") {
            return ("DiT video (es. Wan 2.1)", false, "ComfyUI → models/loras/ o models/diffusion_models/")
        }
        if sample.contains("model.layers.") && sample.contains("self_attn") {
            return ("LLM famiglia Llama/Qwen/Mistral", true, "")
        }
        if sample.contains("transformer.h.") {
            return ("LLM famiglia GPT", true, "")
        }
        if sample.contains("text_model.encoder.layers") {
            return ("Text encoder CLIP (componente di modelli immagine)", false, "ComfyUI → models/text_encoders/")
        }
        return ("non riconosciuta", false, "identificala dai nomi dei tensori")
    }

    private static func formatParams(_ count: Int64) -> String {
        switch count {
        case 1_000_000_000...: String(format: "%.1fB", Double(count) / 1e9)
        case 1_000_000...: String(format: "%.0fM", Double(count) / 1e6)
        default: "\(count)"
        }
    }
}
