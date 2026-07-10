import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import QwenLocalCore
import Tokenizers

/// Registra le tool call eseguite durante il test e fa da fusibile:
/// oltre il limite interrompe il test invece di lasciare il modello in loop.
actor Recorder {
    private(set) var calls: [String] = []
    let maxCalls = 12

    func add(_ name: String) throws {
        calls.append(name)
        if calls.count > maxCalls {
            throw AgentToolbox.ToolboxError("Troppe tool call (\(calls.count)): il modello è in loop, test interrotto.")
        }
    }
}

/// Smoke test end-to-end del loop agentico: scarica il Qwen più piccolo,
/// gli dà i tool del workspace e verifica lettura e scrittura di file reali.
///
/// Uso: ./run.sh smoke (usa Qwen3 1.7B: lo 0.6B è troppo piccolo e va in loop)
@main
struct Smoke {
    static func main() async {
        // Fusibile globale: se il test non finisce in 6 minuti, esce.
        Task.detached {
            try? await Task.sleep(for: .seconds(360))
            print("SMOKE FAIL: timeout globale (6 minuti)")
            exit(2)
        }
        do {
            if ProcessInfo.processInfo.environment["QWEN_SMOKE_RAG"] == "1" {
                try await ragCheck()
            } else {
                try await run()
            }
        } catch {
            print("SMOKE FAIL: \(error)")
            exit(1)
        }
    }

    /// Verifica il percorso di embedding usato dal RAG dell'app: carica e5-small,
    /// calcola gli embedding di frasi simili/diverse e controlla le similarità.
    static func ragCheck() async throws {
        print("Scarico/carico multilingual-e5-small…")
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: EmbedderRegistry.multilingual_e5_small
        )
        MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
        print("Embedder pronto.")

        let texts = [
            "query: come si prepara la pasta alla carbonara",
            "passage: La carbonara si fa con guanciale, uova, pecorino e pepe nero.",
            "passage: Il fatturato del terzo trimestre è cresciuto del 12 per cento.",
        ]
        let vectors: [[Float]] = try await container.perform { context in
            let inputs = texts.map { text in
                Array(context.tokenizer.encode(text: text, addSpecialTokens: true).prefix(510))
            }
            let eos = context.tokenizer.eosTokenId ?? 0
            let maxLength = inputs.reduce(16) { max($0, $1.count) }
            let padded = stacked(
                inputs.map { MLXArray($0 + Array(repeating: eos, count: maxLength - $0.count)) })
            let mask = padded .!= MLXArray(eos)
            let output = context.pooling(
                context.model(
                    padded, positionIds: nil, tokenTypeIds: MLXArray.zeros(like: padded),
                    attentionMask: mask),
                mask: mask, normalize: true, applyLayerNorm: false)
            output.eval()
            return output.map { $0.asArray(Float.self) }
        }

        func dot(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        }
        let similar = dot(vectors[0], vectors[1])
        let different = dot(vectors[0], vectors[2])
        print(String(format: "similarità (carbonara ↔ ricetta):  %.3f", similar))
        print(String(format: "similarità (carbonara ↔ bilancio): %.3f", different))
        print("dimensioni embedding: \(vectors[0].count)")

        let pass = similar > different && similar > 0.7
        print(pass ? "=== RAG CHECK PASS ===" : "=== RAG CHECK FAIL ===")
        exit(pass ? 0 : 1)
    }

    static func run() async throws {
        let ws = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen-smoke-\(ProcessInfo.processInfo.processIdentifier)")
        try FileManager.default.createDirectory(at: ws, withIntermediateDirectories: true)
        try "Il codice segreto è ARANCIA-42.\n".write(
            to: ws.appendingPathComponent("hello.txt"), atomically: true, encoding: .utf8)

        let modelID = ProcessInfo.processInfo.environment["QWEN_SMOKE_MODEL"]
            ?? "mlx-community/Qwen3-4B-4bit"
        print("Workspace: \(ws.path)")
        print("Scarico/carico \(modelID)…")

        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(id: modelID)
        )
        // Stessi limiti dell'app: cache contenuta e tetto rigido di memoria GPU,
        // così un loop di generazione fallisce invece di saturare la RAM.
        MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
        MLX.GPU.set(memoryLimit: Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.6), relaxed: false)
        print("Modello pronto.")

        let toolbox = AgentToolbox(workspace: ws)
        let recorder = Recorder()

        let session = ChatSession(
            container,
            instructions: """
                You are a coding agent working inside this workspace directory: \(ws.path)
                Use the available tools (read_file, write_file, edit_file, list_directory, \
                glob, grep, bash) to accomplish the user's task. File paths are relative to \
                the workspace root. Call a tool at most once per task. Keep answers short.
                """,
            generateParameters: GenerateParameters(maxTokens: 1024, temperature: 0.1),
            additionalContext: ["enable_thinking": false],
            tools: toolbox.specs,
            toolDispatch: { call in
                print("  → tool: \(call.function.name)")
                try await recorder.add(call.function.name)
                let result = try await toolbox.dispatch(call)
                print("  ← \(result.prefix(100).replacingOccurrences(of: "\n", with: " ⏎ "))")
                return result
            }
        )

        print("\n[Test 0] Chat semplice senza tool")
        let answer0 = try await session.respond(
            to: "What is 2+2? Reply with just the number.")
        print("Risposta: \(answer0)")

        print("\n[Test 1] Lettura file via tool")
        let answer1 = try await session.respond(
            to: "Read the file hello.txt with the read_file tool and report the secret code it contains.")
        print("Risposta: \(answer1)")

        print("\n[Test 2] Scrittura file via tool")
        let answer2 = try await session.respond(
            to: "Create a new file named result.txt containing exactly the single word CIAO. Use the write_file tool.")
        print("Risposta: \(answer2)")

        let calls = await recorder.calls
        let readCalled = calls.contains("read_file")
        let codeReported = answer1.contains("ARANCIA")
        let fileWritten =
            (try? String(contentsOf: ws.appendingPathComponent("result.txt"), encoding: .utf8))?
            .contains("CIAO") ?? false

        print("\n=== RISULTATI ===")
        print("tool call eseguite: \(calls.joined(separator: ", "))")
        print("read_file chiamato:  \(readCalled ? "PASS" : "FAIL")")
        print("codice riportato:    \(codeReported ? "PASS" : "FAIL")")
        print("result.txt scritto:  \(fileWritten ? "PASS" : "FAIL")")

        try? FileManager.default.removeItem(at: ws)
        exit(readCalled && codeReported && fileWritten ? 0 : 1)
    }
}
