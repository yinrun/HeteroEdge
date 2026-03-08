#!/usr/bin/env bash
set -euo pipefail

export ANDROID_NDK="${ANDROID_NDK:-/Users/yinrun/Library/Android/sdk/ndk/29.0.13113456}"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/Users/yinrun/Workspace/qairt/2.42.0.251225}"

cmake -S . -B build/android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/android -j
