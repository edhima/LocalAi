// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "LocalAi",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        // Logica indipendente dall'interfaccia: toolbox dell'agente e catalogo modelli.
        .target(
            name: "QwenLocalCore",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            path: "Sources/QwenLocalCore"
        ),
        // L'app macOS.
        .executableTarget(
            name: "LocalAi",
            dependencies: [
                "QwenLocalCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/LocalAi"
        ),
        // Smoke test da riga di comando: modello piccolo + loop agentico con tools.
        .executableTarget(
            name: "QwenSmoke",
            dependencies: [
                "QwenLocalCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/QwenSmoke"
        ),
    ]
)
