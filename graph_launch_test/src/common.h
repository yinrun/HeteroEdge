#pragma once
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

// ── Statistics ───────────────────────────────────────────────────────────────
struct Stats {
  double min, p50, avg, p99, max;
};

inline Stats compute_stats(std::vector<double>& data) {
  Stats s = {};
  if (data.empty()) return s;
  std::sort(data.begin(), data.end());
  s.min = data.front();
  s.max = data.back();
  s.p50 = data[data.size() / 2];
  s.p99 = data[std::min((size_t)(data.size() * 0.99), data.size() - 1)];
  s.avg = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  return s;
}

// ── Timing ───────────────────────────────────────────────────────────────────
inline double now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline double now_us() { return now_seconds() * 1e6; }

// ── ION buffer ───────────────────────────────────────────────────────────────
struct IonBuffer {
  void*  ptr  = nullptr;
  int    fd   = -1;
  size_t size = 0;
};

// ── rpcmem helpers ───────────────────────────────────────────────────────────
struct RpcMemApi {
  void* libHandle = nullptr;
  void* (*alloc)(int, int, int) = nullptr;
  void  (*freeMem)(void*)       = nullptr;
  int   (*toFd)(void*)          = nullptr;
};

inline RpcMemApi& getRpcMemApi() {
  static RpcMemApi api;
  static bool init = false;
  if (!init) {
    init = true;
    const char* candidates[] = {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so",
        "/system/lib64/libcdsprpc.so",
        nullptr};
    for (const char** p = candidates; *p && !api.libHandle; ++p)
      api.libHandle = dlopen(*p, RTLD_LAZY | RTLD_LOCAL);
    if (api.libHandle) {
      api.alloc   = reinterpret_cast<void*(*)(int,int,int)>(dlsym(api.libHandle, "rpcmem_alloc"));
      api.freeMem = reinterpret_cast<void(*)(void*)>(dlsym(api.libHandle, "rpcmem_free"));
      api.toFd    = reinterpret_cast<int(*)(void*)>(dlsym(api.libHandle, "rpcmem_to_fd"));
    }
  }
  return api;
}

constexpr int RPCMEM_HEAP_ID_SYSTEM = 25;

inline bool allocIonBuffer(size_t size, uint8_t fillValue, IonBuffer& out) {
  auto& rpc = getRpcMemApi();
  if (!rpc.alloc || !rpc.toFd) return false;
  out.ptr = rpc.alloc(RPCMEM_HEAP_ID_SYSTEM, 0, static_cast<int>(size));
  if (!out.ptr) return false;
  out.size = size;
  std::memset(out.ptr, fillValue, size);
  out.fd = rpc.toFd(out.ptr);
  if (out.fd < 0) {
    rpc.freeMem(out.ptr);
    out.ptr = nullptr;
    return false;
  }
  return true;
}

inline void freeIonBuffer(IonBuffer& buf) {
  if (buf.ptr) {
    auto& rpc = getRpcMemApi();
    if (rpc.freeMem) rpc.freeMem(buf.ptr);
    buf.ptr = nullptr;
    buf.fd  = -1;
    buf.size = 0;
  }
}

// ── FP16 conversion ─────────────────────────────────────────────────────────
inline uint16_t float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint16_t sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mant >> 13);
}

inline void fillRandomFp16(void* ptr, size_t count, uint32_t seed = 42) {
  uint16_t* p = reinterpret_cast<uint16_t*>(ptr);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;  // LCG
    float v = 0.01f * ((float)(s >> 16) / 65536.0f - 0.5f);  // small values
    p[i] = float_to_half(v);
  }
}
