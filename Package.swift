// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SwiftMatrix",
    platforms: [.macOS(.v13), .iOS(.v16), .tvOS(.v16), .watchOS(.v9)],
    products: [
        .library(
            name: "SwiftMatrix",
            targets: ["SwiftMatrix"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Jounce/Surge.git", from: "2.3.2"),
        .package(url: "https://github.com/jjjkkkjjj/Matft.git", from: "0.3.3"),
    ],
    targets: [
        .target(name: "SwiftMatrix"),
        .testTarget(
            name: "SwiftMatrixTests",
            dependencies: ["SwiftMatrix"]
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: [
                "SwiftMatrix",
                .product(name: "Surge", package: "Surge"),
                .product(name: "Matft", package: "Matft"),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        ),
    ]
)
