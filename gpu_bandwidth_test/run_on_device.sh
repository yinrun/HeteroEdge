#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="/data/local/tmp/gpu_bandwidth_test"

adb shell "mkdir -p ${DEVICE_DIR}/kernels"

adb push build/android/gpu_bandwidth_test "${DEVICE_DIR}/"
adb push kernels/vector_copy.cl "${DEVICE_DIR}/kernels/"

adb shell "chmod 755 ${DEVICE_DIR}/gpu_bandwidth_test"

adb shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=/vendor/lib64:\$LD_LIBRARY_PATH && \
  ./gpu_bandwidth_test $*"
