/*!
 *  Copyright (c) 2024 by Contributors if not otherwise specified
 * \file videotoolbox_device_api.cc
 * \brief VideoToolbox device API implementation for macOS Metal devices
 */

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <decord/runtime/registry.h>
#include <decord/runtime/device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace decord {
namespace runtime {

class VideoToolboxDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DECORDContext ctx) final {
    // VideoToolbox handles device selection internally
    // No explicit device setting needed for Metal/VideoToolbox
  }

  void GetAttr(DECORDContext ctx, DeviceAttrKind kind, DECORDRetValue* rv) final {
#ifdef __APPLE__
    switch (kind) {
      case kExist: {
        // VideoToolbox is available on macOS
        *rv = 1;
        break;
      }
      case kMaxThreadsPerBlock: {
        // Typical Metal threadgroup size
        *rv = 256;
        break;
      }
      case kWarpSize: {
        // Metal SIMD width
        *rv = 32;
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        // Typical Metal threadgroup memory
        *rv = 16384;
        break;
      }
      case kComputeVersion: {
        // VideoToolbox version
        *rv = std::string("1.0");
        break;
      }
      case kDeviceName: {
        *rv = std::string("VideoToolbox GPU");
        break;
      }
      case kMaxClockRate: {
        // Default clock rate
        *rv = 1000;
        break;
      }
      case kMultiProcessorCount: {
        // Approximate compute units
        *rv = 8;
        break;
      }
      case kMaxThreadDimensions: {
        // Default thread dimensions
        *rv = std::string("256x256x64");
        break;
      }
      default:
        LOG(FATAL) << "unknown device attribute type " << kind;
    }
#else
    // Non-Apple platforms
    *rv = 0;
#endif
  }

  void* AllocDataSpace(DECORDContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DECORDType type_hint) final {
    // Use aligned malloc for simplicity
    return aligned_alloc(alignment, nbytes);
  }

  void FreeDataSpace(DECORDContext ctx, void* ptr) final {
    if (ptr) {
      free(ptr);
    }
  }

  void* AllocWorkspace(DECORDContext ctx, size_t size, DECORDType type_hint) final {
    return AllocDataSpace(ctx, size, kAllocAlignment, type_hint);
  }

  void FreeWorkspace(DECORDContext ctx, void* data) final {
    FreeDataSpace(ctx, data);
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t num_bytes,
                      DECORDContext ctx_from,
                      DECORDContext ctx_to,
                      DECORDType type_hint,
                      DECORDStreamHandle stream) final {
    // Simple memory copy for now
    // In a full implementation, this would handle Metal buffer copies
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset,
           num_bytes);
  }

  void StreamSync(DECORDContext ctx, DECORDStreamHandle stream) final {
    // Metal command buffer synchronization would go here
    // For now, this is a no-op
  }
};

DECORD_REGISTER_GLOBAL("device_api.metal")
.set_body([](DECORDArgs args, DECORDRetValue *ret) {
    DeviceAPI* ptr = new VideoToolboxDeviceAPI();
    *ret = ptr;
  });

}  // namespace runtime
}  // namespace decord