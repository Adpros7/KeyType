#!/bin/zsh
set -euo pipefail

# Ensure the gitignored llama.cpp xcframework required by ModelRuntime is present.
#
# Usage:
#   ./Scripts/bootstrap-llama.sh
#   ./Scripts/bootstrap-llama.sh --force

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
SCRIPT_NAME="./Scripts/$(basename "$0")"

LLAMA_BUILD="b9402"
LLAMA_ZIP="llama-$LLAMA_BUILD-xcframework.zip"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/$LLAMA_ZIP"
VENDOR_DIR="$REPO_PATH/Packages/ModelRuntime/Vendor"
FRAMEWORK_DIR="$VENDOR_DIR/llama.xcframework"
TMP_DIR="$VENDOR_DIR/.tmp-$LLAMA_BUILD"
FORCE=0

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [--force]

Downloads the llama.cpp $LLAMA_BUILD xcframework into:
  $FRAMEWORK_DIR

The Vendor directory is gitignored; this is a local build prerequisite.
EOF
    exit "${1:-1}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)
            FORCE=1
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

framework_is_valid() {
    [ -f "$FRAMEWORK_DIR/Info.plist" ] &&
    find "$FRAMEWORK_DIR" -path "*/llama.framework/llama" -type f -perm +111 -print -quit | grep -q .
}

if [ "$FORCE" -eq 0 ] && framework_is_valid; then
    echo "llama.xcframework already present: $FRAMEWORK_DIR"
    exit 0
fi

mkdir -p "$VENDOR_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Downloading llama.cpp $LLAMA_BUILD xcframework..."
curl -L --fail --retry 3 --output "$TMP_DIR/$LLAMA_ZIP" "$LLAMA_URL"

echo "Unpacking llama.xcframework..."
ditto -x -k "$TMP_DIR/$LLAMA_ZIP" "$TMP_DIR/unpacked"

FOUND_FRAMEWORK="$(find "$TMP_DIR/unpacked" -name llama.xcframework -type d -maxdepth 4 -print -quit)"
if [ -z "$FOUND_FRAMEWORK" ]; then
    echo "Error: downloaded archive did not contain llama.xcframework" >&2
    exit 1
fi

rm -rf "$FRAMEWORK_DIR"
ditto "$FOUND_FRAMEWORK" "$FRAMEWORK_DIR"
rm -rf "$TMP_DIR"

if ! framework_is_valid; then
    echo "Error: installed llama.xcframework does not contain a usable binary artifact" >&2
    exit 1
fi

echo "Installed llama.xcframework: $FRAMEWORK_DIR"
