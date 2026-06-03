#!/bin/zsh
set -euo pipefail

# Build KeyType locally without archive/export/notarization/appcast work.
#
# Usage:
#   ./Scripts/build-local.sh
#   ./Scripts/build-local.sh --release
#   ./Scripts/build-local.sh --clean --open

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
SCRIPT_NAME="./Scripts/$(basename "$0")"

APP_NAME="KeyType"
WORKSPACE="$REPO_PATH/KeyType.xcworkspace"
SCHEME="KeyType"
CONFIGURATION="Debug"
BUILD_ROOT="$REPO_PATH/build/local"
DERIVED_DATA="$BUILD_ROOT/DerivedData"
OUTPUT_DIR="$BUILD_ROOT/App"
DMG_DIR="$BUILD_ROOT/DMG"
DMG_STAGING_DIR="$BUILD_ROOT/DMGStaging"
DMG_PATH="$DMG_DIR/$APP_NAME.dmg"
CLEAN=0
OPEN_APP=0

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [--debug|--release] [--clean] [--open]

Builds $APP_NAME locally and packages it into a simple unsigned DMG. No archive, export,
notarization, Sparkle signing, appcast update, or GitHub release is performed.

Output:
  $OUTPUT_DIR/$APP_NAME.app
  $DMG_PATH
EOF
    exit "${1:-1}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --debug)
            CONFIGURATION="Debug"
            ;;
        --release)
            CONFIGURATION="Release"
            ;;
        --clean)
            CLEAN=1
            ;;
        --open)
            OPEN_APP=1
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            usage
            ;;
    esac
    shift
done

mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR" "$DMG_DIR"

if [ "$CLEAN" -eq 1 ]; then
    echo "Cleaning local build output..."
    rm -rf "$DERIVED_DATA" "$OUTPUT_DIR/$APP_NAME.app" "$DMG_PATH" "$DMG_STAGING_DIR"
fi

"$SCRIPT_DIR/bootstrap-llama.sh"

echo "Building $APP_NAME ($CONFIGURATION) locally..."
xcodebuild build \
    -workspace "$WORKSPACE" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$DERIVED_DATA" \
    -destination "platform=macOS" \
    CODE_SIGNING_ALLOWED=NO \
    -quiet

BUILT_APP="$DERIVED_DATA/Build/Products/$CONFIGURATION/$APP_NAME.app"
if [ ! -d "$BUILT_APP" ]; then
    echo "Error: built app not found at $BUILT_APP" >&2
    exit 1
fi

rm -rf "$OUTPUT_DIR/$APP_NAME.app"
ditto "$BUILT_APP" "$OUTPUT_DIR/$APP_NAME.app"

echo "Built app: $OUTPUT_DIR/$APP_NAME.app"

echo "Creating local DMG..."
rm -rf "$DMG_STAGING_DIR" "$DMG_PATH"
mkdir -p "$DMG_STAGING_DIR"
ditto "$OUTPUT_DIR/$APP_NAME.app" "$DMG_STAGING_DIR/$APP_NAME.app"
ln -s /Applications "$DMG_STAGING_DIR/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGING_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH" >/dev/null

rm -rf "$DMG_STAGING_DIR"
echo "Built DMG: $DMG_PATH"

if [ "$OPEN_APP" -eq 1 ]; then
    open "$DMG_PATH"
fi
