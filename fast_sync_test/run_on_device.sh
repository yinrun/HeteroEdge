#!/usr/bin/env bash
set -euo pipefail
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"

DEVICE_DIR="/data/local/tmp/fast_sync_test"
LIB_DIR="${DEVICE_DIR}/lib"
HTP_DIR="${DEVICE_DIR}/htp"

adb shell "mkdir -p ${DEVICE_DIR}/kernels ${LIB_DIR} ${HTP_DIR}"

adb push build/android/fast_sync_test "${DEVICE_DIR}/"
adb push kernels/rmsnorm.cl "${DEVICE_DIR}/kernels/"

# Custom FlagPoll RMSNorm op package (for NPU_POLL mode)
CUSTOM_OP_DIR="custom_op/build"
if [ -f "${CUSTOM_OP_DIR}/aarch64-android/libQnnHtpFlagPollRmsNormOpPackage.so" ]; then
  adb push "${CUSTOM_OP_DIR}/aarch64-android/libQnnHtpFlagPollRmsNormOpPackage.so" "${LIB_DIR}/"
  echo "Pushed custom op (aarch64) to ${LIB_DIR}"
fi
if [ -f "${CUSTOM_OP_DIR}/hexagon-v81/libQnnHtpFlagPollRmsNormOpPackage.so" ]; then
  adb push "${CUSTOM_OP_DIR}/hexagon-v81/libQnnHtpFlagPollRmsNormOpPackage.so" "${HTP_DIR}/"
  echo "Pushed custom op (hexagon-v81) to ${HTP_DIR}"
fi

# QNN runtime libraries
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB_DIR}/"

# Hexagon V81 skel libraries
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP_DIR}/"

adb shell "chmod 755 ${DEVICE_DIR}/fast_sync_test"

adb shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${LIB_DIR}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP_DIR} && \
  ./fast_sync_test $*"
