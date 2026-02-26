// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SwiftMatrix",
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
