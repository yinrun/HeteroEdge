#!/usr/bin/env bash
set -euo pipefail

# Build custom FlagPoll RMSNorm HTP Op Package
# Requires: QNN_SDK_ROOT, HEXAGON_SDK_ROOT, ANDROID_NDK_ROOT

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/Users/yinrun/Workspace/qairt/2.42.0.251225}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/Users/yinrun/Workspace/qairt/Hexagon_SDK/6.5.0.0}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/Users/yinrun/Library/Android/sdk/ndk/29.0.13113456}"

echo "=== Building FlagPoll RMSNorm Custom Op Package ==="
echo "QNN_SDK_ROOT:     ${QNN_SDK_ROOT}"
echo "HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}"
echo "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}"
echo ""

cd "$(dirname "$0")"

make clean
make all

echo ""
echo "Build complete:"
ls -la build/hexagon-v81/*.so 2>/dev/null || echo "  (no hexagon-v81 output)"
ls -la build/aarch64-android/*.so 2>/dev/null || echo "  (no aarch64-android output)"
