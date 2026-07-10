#!/bin/zsh
# Compila LocalAi e assembla il bundle LocalAi.app.
#
# Nota: serve xcodebuild (non `swift build`) perché gli shader Metal di MLX
# non vengono compilati dalla CLI di SwiftPM. Serve anche il Metal Toolchain
# (una tantum: xcodebuild -downloadComponent MetalToolchain).
#
# Uso:
#   ./run.sh            # compila (Release), crea LocalAi.app e la apre
#   ./run.sh build      # compila e crea LocalAi.app soltanto
#   ./run.sh smoke      # compila e lancia lo smoke test CLI dell'agente
set -e
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")"

MODE="${1:-app}"
CONFIG=Release

if [[ "$MODE" == "smoke" ]]; then
  echo "Compilo QwenSmoke ($CONFIG)…"
  xcodebuild build -scheme QwenSmoke -configuration "$CONFIG" \
    -destination 'platform=macOS' -derivedDataPath .xcodebuild \
    -skipMacroValidation -skipPackagePluginValidation -quiet
  exec ".xcodebuild/Build/Products/$CONFIG/QwenSmoke"
fi

echo "Compilo LocalAi ($CONFIG)…"
xcodebuild build -scheme LocalAi -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath .xcodebuild \
  -skipMacroValidation -skipPackagePluginValidation -quiet

PRODUCTS=".xcodebuild/Build/Products/$CONFIG"
APP="LocalAi.app"

# --- Runtime Python integrato (CPython standalone ricollocabile + mlx-lm) ---
# Scaricato e preparato una sola volta in .tooling/python, poi copiato nel
# bundle: l'app è autonoma, la disinstallazione è "cestina LocalAi.app".
PBS_TAG="20251217"
PBS_PY="3.12.12"
RUNTIME_DIR=".tooling/python"
RUNTIME_MARKER="$RUNTIME_DIR/.localai-ready-$PBS_PY-$PBS_TAG"

if [[ ! -f "$RUNTIME_MARKER" ]]; then
  echo "Preparo il runtime Python integrato ($PBS_PY, una tantum)…"
  rm -rf "$RUNTIME_DIR"
  mkdir -p .tooling
  TARBALL=".tooling/cpython.tar.gz"
  curl -sL -o "$TARBALL" \
    "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/cpython-$PBS_PY+$PBS_TAG-aarch64-apple-darwin-install_only.tar.gz"
  tar -xzf "$TARBALL" -C .tooling   # estrae .tooling/python
  rm -f "$TARBALL"
  echo "Installo mlx-lm nel runtime…"
  "$RUNTIME_DIR/bin/python3" -m pip install --quiet --upgrade pip
  # transformers pinnato: la 5.13 rompe l'import di mlx-lm 0.31.x
  "$RUNTIME_DIR/bin/python3" -m pip install --quiet mlx-lm "transformers~=5.12.0"
  # dimagrimento: cache e test non servono a runtime
  find "$RUNTIME_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$RUNTIME_DIR/lib/python3.12/test" 2>/dev/null || true
  "$RUNTIME_DIR/bin/python3" -m mlx_lm.lora --help > /dev/null
  touch "$RUNTIME_MARKER"
  echo "Runtime pronto."
fi

echo "Assemblo $APP…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PRODUCTS/LocalAi" "$APP/Contents/MacOS/LocalAi"
# I bundle di risorse SwiftPM vanno in Contents/Resources (dove guardano
# Bundle.module e il loader di MLX via mainBundle.resourceURL).
for bundle in "$PRODUCTS"/*.bundle; do
  [[ -e "$bundle" ]] && cp -R "$bundle" "$APP/Contents/Resources/"
done
# Cintura e bretelle: MLX cerca anche "mlx.metallib" accanto all'eseguibile.
METALLIB="$PRODUCTS/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
[[ -f "$METALLIB" ]] && cp "$METALLIB" "$APP/Contents/MacOS/mlx.metallib"
# Runtime Python + mlx-lm integrati: l'app non dipende da nulla di esterno.
cp -R "$RUNTIME_DIR" "$APP/Contents/Resources/python"
# Icona dell'app
[[ -f "Assets/AppIcon.icns" ]] && cp "Assets/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>LocalAi</string>
    <key>CFBundleIdentifier</key>
    <string>dev.eridon.localai</string>
    <key>CFBundleName</key>
    <string>LocalAi</string>
    <key>CFBundleDisplayName</key>
    <string>LocalAi</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>© 2026 Eridon Dhima — e.dhima@alpha-soft.al</string>
</dict>
</plist>
PLIST
# Firma ad-hoc: senza firma macOS moderno rifiuta di lanciare il bundle.
codesign --force --deep --sign - "$APP"

echo "Creato: $APP"

if [[ "$MODE" != "build" ]]; then
  open "$APP"
fi
