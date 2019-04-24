/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cpu_device_api.cc
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <decord/runtime/registry.h>
#include <decord/runtime/device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace decord {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DECORDContext ctx) final {}
  void GetAttr(DECORDContext ctx, DeviceAttrKind kind, DECORDRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(DECORDContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DECORDType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG) || (defined(__ANDROID__) && __ANDROID_API__ < 16)
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 16
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(DECORDContext ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      DECORDContext ctx_from,
                      DECORDContext ctx_to,
                      DECORDType type_hint,
                      DECORDStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset,
           size);
  }

  void StreamSync(DECORDContext ctx, DECORDStreamHandle stream) final {
  }

  void* AllocWorkspace(DECORDContext ctx, size_t size, DECORDType type_hint) final;
  void FreeWorkspace(DECORDContext ctx, void* data) final;

  static const std::shared_ptr<CPUDeviceAPI>& Global() {
    static std::shared_ptr<CPUDeviceAPI> inst =
        std::make_shared<CPUDeviceAPI>();
    return inst;
  }
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() :
      WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(DECORDContext ctx,
                                   size_t size,
                                   DECORDType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void CPUDeviceAPI::FreeWorkspace(DECORDContext ctx, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

DECORD_REGISTER_GLOBAL("device_api.cpu")
.set_body([](DECORDArgs args, DECORDRetValue* rv) {
    DeviceAPI* ptr = CPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace decord
