#!/usr/bin/env bash
set -euo pipefail

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/Users/yinrun/Workspace/qairt/2.42.0.251225}"

DEVICE_DIR="/data/local/tmp/graph_launch_test"
LIB_DIR="${DEVICE_DIR}/lib"
HTP_DIR="${DEVICE_DIR}/htp"

adb shell "mkdir -p ${LIB_DIR} ${HTP_DIR}"

# Push binary
adb push build/android/graph_launch_test "${DEVICE_DIR}/"

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

adb shell "chmod 755 ${DEVICE_DIR}/graph_launch_test"

adb shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${LIB_DIR}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP_DIR} && \
  ./graph_launch_test $*"
