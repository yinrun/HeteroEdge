#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_SDK_ROOT:-}" && -d "${ANDROID_SDK_ROOT}/ndk" ]]; then
    ANDROID_NDK="$(ls -1 "${ANDROID_SDK_ROOT}/ndk" | sort -V | tail -n 1)"
    ANDROID_NDK="${ANDROID_SDK_ROOT}/ndk/${ANDROID_NDK}"
    export ANDROID_NDK
  else
    echo "ANDROID_NDK is not set. Set ANDROID_SDK_ROOT to auto-pick NDK."
    exit 1
  fi
fi

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "QNN_SDK_ROOT is not set"
  exit 1
fi

cmake -S . -B build/android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/android -j
