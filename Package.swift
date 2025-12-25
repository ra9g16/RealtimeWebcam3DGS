// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "RealtimeWebcam3DGS",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "RealtimeWebcam3DGS",
            targets: ["RealtimeWebcam3DGS"]
        ),
    ],
    dependencies: [
        .package(path: ".."),  // MetalSplatter
    ],
    targets: [
        .executableTarget(
            name: "RealtimeWebcam3DGS",
            dependencies: [
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
    ]
)
