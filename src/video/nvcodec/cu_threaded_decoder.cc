/*!
 *  Copyright (c) 2019 by Contributors
 * \file cu_decoder.cc
 * \brief NVCUVID based decoder impl
 */

#include "cu_threaded_decoder.h"

namespace decord {
namespace cuda {

CUThreadedDecoder::CUThreadedDecoder(int device_id, const AVCodecContext *dec_ctx) 
    : device_id_(device_id), device_{}, ctx_{}, parser_{}, decoder_{}, 
    occupied_frames_(32), // 32 is cuvid's max number of decode surfaces
    pkt_queue_{}, buffer_queue_{}, frame_queue_{}, run_(false) {
    if (!dec_ctx_) {
        LOG(FATAL) << "Invalid CodecContext!";
    }
    
    if (!CUDA_CHECK_CALL(cuInit(0))) {
        LOG(FATAL) << "Unable to initial cuda driver. Is the kernel module installed?";
    }

    if (!CUDA_CHECK_CALL(cuDeviceGet(&device_, device_id_))) {
        LOG(FATAL) << "Problem getting device info for device "
                  << device_id_ << ", not initializing VideoDecoder\n";
        return;
    }

    char device_name[100];
    if (!CUDA_CHECK_CALL(cuDeviceGetName(device_name, 100, device_))) {
        LOG(FATAL) << "Problem getting device name for device "
                  << device_id_ << ", not initializing VideoDecoder\n";
        return;
    }
    DLOG(INFO) << "Using device: " << device_name;

    try {
        auto nvml_ret = nvmlInit();
        if (nvml_ret != NVML_SUCCESS) {
            LOG(FATAL) << "nvmlInit returned error " << nvml_ret;
        }
        char nvmod_version_string[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        nvml_ret = nvmlSystemGetDriverVersion(nvmod_version_string,
                                              sizeof(nvmod_version_string));
        if (nvml_ret != NVML_SUCCESS) {
            LOG(FATAL) << "nvmlSystemGetDriverVersion returned error " << nvml_ret;
        }
        auto nvmod_version = std::stof(nvmod_version_string);
        if (nvmod_version < 384.0f) {
            LOG(INFO) << "Older kernel module version " << nvmod_version
                        << " so using the default stream."
                        << std::endl;
            use_default_stream();
        } else {
            LOG(INFO) << "Kernel module version " << nvmod_version
                        << ", so using our own stream."
                        << std::endl;
        }
    } catch(const std::exception& e) {
        LOG(INFO) << "Unable to get nvidia kernel module version from NVML, "
                    << "conservatively assuming it is an older version.\n"
                    << "The error was: " << e.what()
                    << std::endl;
        use_default_stream();
    }

    parser_ = CUVideoParser(dec_ctx->codec, this, 20, dec_ctx->codecpar->extradata,
                            dec_ctx->codecpar->extradata_size);
    if (!parser_.Initialized()) {
        LOG(FATAL) << "Problem creating video parser";
        return;
    }

}  // namespace cuda
}  // namespace decord