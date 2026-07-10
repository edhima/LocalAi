import SwiftUI

/// Palette e tipografia in stile terminale Claude Code: tema scuro, font mono,
/// arancione come colore d'accento.
enum Theme {
    // Sfondi
    static let background = Color(red: 0.149, green: 0.145, blue: 0.141)  // #262624
    static let panel = Color(red: 0.188, green: 0.184, blue: 0.176)       // #30302D
    static let inputBackground = Color(red: 0.165, green: 0.161, blue: 0.153)

    // Testi
    static let text = Color(red: 0.961, green: 0.956, blue: 0.937)        // #F5F4EF
    static let secondary = Color(white: 0.64)
    static let dim = Color(white: 0.44)

    // Accenti
    static let accent = Color(red: 0.851, green: 0.467, blue: 0.341)      // #D97757
    static let green = Color(red: 0.42, green: 0.75, blue: 0.48)
    static let red = Color(red: 0.91, green: 0.45, blue: 0.42)
    static let orange = Color(red: 0.95, green: 0.68, blue: 0.35)
    static let border = Color(white: 0.32)

    static func mono(_ size: CGFloat = 13, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}
