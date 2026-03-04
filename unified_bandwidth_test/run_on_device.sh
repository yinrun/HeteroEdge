#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "QNN_SDK_ROOT is not set"
  exit 1
fi

DEVICE_DIR="/data/local/tmp/unified_bandwidth_test"
LIB_DIR="${DEVICE_DIR}/lib"
HTP_DIR="${DEVICE_DIR}/htp"

adb shell "mkdir -p ${DEVICE_DIR}/kernels ${LIB_DIR} ${HTP_DIR}"

# Push binary and kernel
adb push build/android/unified_bandwidth_test "${DEVICE_DIR}/"
adb push kernels/element_add.cl "${DEVICE_DIR}/kernels/"

# Push QNN ARM64 libraries
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB_DIR}/"

# Push Hexagon V81 libraries
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP_DIR}/"

adb shell "chmod 755 ${DEVICE_DIR}/unified_bandwidth_test"

# LD_LIBRARY_PATH: QNN libs + vendor OpenCL
# ADSP_LIBRARY_PATH: Hexagon DSP libraries
adb shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${LIB_DIR}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP_DIR} && \
  ./unified_bandwidth_test $*"
