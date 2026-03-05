#=============================================================================
#  Custom RMSNorm HTP Op Package - Makefile
#  Targets: hexagon-v81 (SM8850), aarch64-android, x86_64-linux-clang
#=============================================================================

# Required environment variables:
#   QNN_SDK_ROOT     - QAIRT SDK root (e.g. /path/to/qairt/2.42.0.251225)
#   HEXAGON_SDK_ROOT - Hexagon SDK root (e.g. /path/to/Hexagon_SDK/6.5.0.0)
# Optional:
#   ANDROID_NDK_ROOT - Android NDK root (required for aarch64 target)

ifneq ($(MAKECMDGOALS),clean)
ifndef QNN_SDK_ROOT
$(error "ERROR: QNN_SDK_ROOT is not set")
endif
ifndef HEXAGON_SDK_ROOT
$(error "ERROR: HEXAGON_SDK_ROOT is not set")
endif

QNN_INCLUDE ?= $(QNN_SDK_ROOT)/include/QNN
QNN_TARGET_LIB ?= $(QNN_SDK_ROOT)/lib/aarch64-android

ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path does not exist: $(QNN_INCLUDE)")
endif
endif

# Package configuration
PACKAGE_NAME = rmsnorm.HvxOpPackage
LIBRARY_NAME = libQnnHtpRmsNormOpPackage.so
WORK := build

# Source files
OP_SOURCES = $(wildcard *.cpp)
OP_OBJS = $(patsubst %.cpp,%.o,$(OP_SOURCES))

# Headers
HFILES = $(wildcard $(QNN_INCLUDE)/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/core/*.h)

# Common flags
COMMON_CXX_FLAGS = -std=c++17 -I$(QNN_INCLUDE) -fPIC -Wall -Wreorder -Wno-missing-braces
COMMON_CXX_FLAGS += -Wno-unused-function -Wno-format -Wno-unused-command-line-argument
COMMON_CXX_FLAGS += -fvisibility=default -stdlib=libc++
COMMON_CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"
COMMON_CXX_FLAGS += -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

#============================================================================================================
# Hexagon V81 toolchain
HEXAGON_TOOLS_VERSION := 19.0.07
HEXAGON_TOOLS_DIR := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools
HEXAGON_CXX := $(HEXAGON_TOOLS_DIR)/bin/hexagon-clang++

HEXAGON_CXX_FLAGS := $(COMMON_CXX_FLAGS) -mhvx -mhvx-length=128B -mhmx -O2
HEXAGON_CXX_FLAGS += -mv81 -DUSE_OS_QURT -DPREPARE_DISABLED
HEXAGON_CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev81/include/qurt
HEXAGON_CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev81/include/posix
HEXAGON_CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/incs
HEXAGON_CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/incs/stddef

#============================================================================================================
# X86 toolchain (for simulation/testing on host)
X86_LIBNATIVE_DIR := $(HEXAGON_TOOLS_DIR)
X86_CXX ?= clang++-9
ifeq ($(shell $(X86_CXX) -v 2>&1 | grep -c "clang version"), 0)
  X86_CXX := clang++
endif
X86_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_DIR)/libnative/include -DUSE_OS_LINUX
X86_CXX_FLAGS := $(COMMON_CXX_FLAGS) $(X86_C_FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof
X86_LDFLAGS := -Wl,--whole-archive -L$(X86_LIBNATIVE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread

#============================================================================================================
# AArch64 Android toolchain
ifdef ANDROID_NDK_ROOT
ARM_CLANG_OPTS := --target=aarch64-none-linux-android21 --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot -stdlib=libc++ -static-libstdc++
AARCH64_CXX := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ $(ARM_CLANG_OPTS)
AARCH64_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_DIR)/libnative/include -DUSE_OS_LINUX -DANDROID
AARCH64_CXX_FLAGS := $(COMMON_CXX_FLAGS) $(AARCH64_C_FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -Wno-unused-private-field -Wno-ignored-qualifiers -Wno-missing-field-initializers
AARCH64_LDFLAGS := -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare
endif

#============================================================================================================
# Build targets

.PHONY: all htp_v81 htp_x86 htp_aarch64 clean

all: htp_v81
ifdef ANDROID_NDK_ROOT
all: htp_aarch64
endif

htp_v81: $(WORK)/hexagon-v81/$(LIBRARY_NAME)

htp_x86: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)

htp_aarch64: $(WORK)/aarch64-android/$(LIBRARY_NAME)

# Create build directories
$(WORK)/hexagon-v81 $(WORK)/x86_64-linux-clang $(WORK)/aarch64-android:
	@mkdir -p $@

# Hexagon V81 rules
$(WORK)/hexagon-v81/%.o: %.cpp | $(WORK)/hexagon-v81
	$(HEXAGON_CXX) $(HEXAGON_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v81/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v81/%,$(OP_OBJS)) | $(HFILES)
	$(HEXAGON_CXX) -fPIC -std=c++17 -g -shared -o $@ $^

# X86 rules
$(WORK)/x86_64-linux-clang/%.o: %.cpp | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(patsubst %,$(WORK)/x86_64-linux-clang/%,$(OP_OBJS)) | $(HFILES)
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(X86_LDFLAGS)

# AArch64 rules
$(WORK)/aarch64-android/%.o: %.cpp | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(patsubst %,$(WORK)/aarch64-android/%,$(OP_OBJS)) | $(HFILES)
	$(AARCH64_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(AARCH64_LDFLAGS)

clean:
	rm -rf $(WORK)
