#!/usr/bin/env bash
set -euo pipefail
export ANDROID_NDK="${ANDROID_NDK:-/home/yinrun/Android/Sdk/android-ndk-r25c}"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"

cmake -S . -B build/android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/android -j
