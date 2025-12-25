import SwiftUI

@main
struct RealtimeWebcam3DGSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.automatic)
        .defaultSize(width: 1200, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {}  // Remove New Window
        }
    }
}
