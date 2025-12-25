import SwiftUI
import CoreGraphics

/// SwiftUI view for displaying the camera preview
struct CameraPreviewView: View {
    @ObservedObject var cameraManager: CameraCaptureManager

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color.black

                // Camera preview image
                if let previewImage = cameraManager.previewImage {
                    Image(decorative: previewImage, scale: 1.0)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "video.slash")
                            .font(.system(size: 48))
                            .foregroundColor(.gray)
                        Text("No camera feed")
                            .foregroundColor(.gray)
                    }
                }

                // Live indicator
                if cameraManager.isCapturing {
                    VStack {
                        HStack {
                            Spacer()
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 8, height: 8)
                                Text("LIVE")
                                    .font(.caption.bold())
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(4)
                            .padding(8)
                        }
                        Spacer()
                    }
                }
            }
        }
        .cornerRadius(8)
    }
}

/// SwiftUI view showing the last captured image
struct LastCapturedView: View {
    @ObservedObject var cameraManager: CameraCaptureManager

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Last Capture")
                .font(.headline)

            if let capturedImage = cameraManager.lastCapturedImage {
                Image(decorative: capturedImage, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .cornerRadius(4)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                    .aspectRatio(16/9, contentMode: .fit)
                    .overlay(
                        Text("No capture yet")
                            .foregroundColor(.gray)
                    )
                    .cornerRadius(4)
            }
        }
    }
}

/// Preview for static image mode
struct StaticImagePreviewView: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color(NSColor.controlBackgroundColor)

                if let image = coordinator.importedImage {
                    // Show imported image
                    Image(decorative: image, scale: 1.0)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    // Placeholder
                    VStack(spacing: 16) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("Import an image to convert to 3DGS")
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        Button(action: {
                            Task {
                                await coordinator.importImage()
                            }
                        }) {
                            Label("Import Image", systemImage: "photo.on.rectangle")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                }

                // Processing overlay
                if coordinator.appState == .generating {
                    Color.black.opacity(0.5)
                    VStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text("Converting to 3DGS...")
                            .foregroundColor(.white)
                            .font(.headline)
                    }
                }
            }
        }
        .cornerRadius(8)
    }
}
