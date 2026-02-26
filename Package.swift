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
    targets: [
        .target(name: "SwiftMatrix"),
        .testTarget(
            name: "SwiftMatrixTests",
            dependencies: ["SwiftMatrix"]
        ),
    ]
)
