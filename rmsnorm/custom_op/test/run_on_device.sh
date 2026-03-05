#!/usr/bin/env bash
set -euo pipefail

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"

DIR="/data/local/tmp/rmsnorm_custom_test"
LIB="${DIR}/lib"
HTP="${DIR}/htp"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_PKG_DIR="${SCRIPT_DIR}/../build"

adb shell "mkdir -p ${DIR} ${LIB} ${HTP}"

# Push test binary
adb push "${SCRIPT_DIR}/build/android/rmsnorm_custom_test" "${DIR}/"

# Push QNN ARM64 libs
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB}/"

# Push Hexagon V81 DSP libs
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP}/"

# Push custom op package libs
if [ -f "${OP_PKG_DIR}/aarch64-android/libQnnHtpRmsNormOpPackage.so" ]; then
  adb push "${OP_PKG_DIR}/aarch64-android/libQnnHtpRmsNormOpPackage.so" "${DIR}/"
fi
if [ -f "${OP_PKG_DIR}/hexagon-v81/libQnnHtpRmsNormOpPackage.so" ]; then
  adb push "${OP_PKG_DIR}/hexagon-v81/libQnnHtpRmsNormOpPackage.so" "${HTP}/"
fi

adb shell "chmod 755 ${DIR}/rmsnorm_custom_test"

echo "=== Running custom RMSNorm test on device ==="
adb shell "cd ${DIR} && \
  export LD_LIBRARY_PATH=${LIB}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP} && \
  ./rmsnorm_custom_test $*"
