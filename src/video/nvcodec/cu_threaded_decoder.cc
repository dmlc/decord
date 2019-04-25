/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
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
    
    if (!CHECK_CUDA_CALL(cuInit(0))) {
        LOG(FATAL) << "Unable to initial cuda driver. Is the kernel module installed?";
    }

    if (!CHECK_CUDA_CALL(cuDeviceGet(&device_, device_id_))) {
        LOG(FATAL) << "Problem getting device info for device "
                  << device_id_ << ", not initializing VideoDecoder\n";
        return;
    }

    char device_name[100];
    if (!CHECK_CUDA_CALL(cuDeviceGetName(device_name, 100, device_))) {
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

}

int CUDAAPI CUThreadedDecoder::HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureSequence_(format);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDecode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureDecode_(pic_params);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDisplay(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureDisplay_(disp_info);
}

int CUThreadedDecoder::HandlePictureSequence_(CUVIDEOFORMAT* format) {
    return decoder_.initialize(format);
}

int CUThreadedDecoder::HandlePictureDecode_(CUVIDPICPARAMS* pic_params) {
    CHECK_GE(pic_pararms->CurrPicIdx, 0);
    CHECK_LT(pic_pararms->CurrPicIdx, permits_.size());
    auto& permit_queue = permits_[pic_pararms->CurrPicIdx];
    bool ret;
    uint8_t tmp;
    if (!run_.load() || !permit_queue.Pop(&tmp)) return 0;
    if (!CHECK_CUDA_CALL(cuvidDecodePicture(decoder_, pic_params))) {
        LOG(FATAL) << "Failed to launch cuvidDecodePicture";
    }
    return 1;
}

int CUThreadedDecoder::HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info) {
    // push to converter
    buffer_queue_.Push(disp_info);
    // finished, send clear msg to allow next decoding
    permits_[disp_info->CurrPicIdx].Push(1);
}

void CUThreadedDecoder::LaunchThread() {
    context_.push();
    while (run_.load()) {
        bool ret;
        AVPacketPtr avpkt = nullptr;
        ret = pkt_queue_.Pop(&avpkt)
        if (!ret) return;

        CUVIDSOURCEDATAPACKET cupkt = {0};

        if (avpkt && avpkt->size) {
            cupkt.payload_size = avpkt->size;
            cupkt.payload = avpkt->data;
            if (avpkt->pts != AV_NOPTS_VALUE) {
                cupkt.flags = CUVID_PKT_TIMESTAMP;
                cupkt.timestamp = avpkt->pts;
            }
        } else {
            cupkt.flags = CUVID_PKT_ENDOFSTREAM;
            // mark as flushing?
        }

        if (!CHECK_CUDA_CALL(cuvidParseVideoData(parser_, &cupkt))) {
            LOG(FATAL) << "Problem decoding packet";
        }
    }
}

void CUThreadedDecoder::ConvertThread() {
    context_.push();
    while (run_.load()) {
        bool ret;
        CUVIDPARSERDISPINFO *disp_info = nullptr;
        ret = buffer_queue_.Pop(&disp_info);
        if (!ret) return;
        CHECK(disp_info != nullptr);
        auto frame = CUMappedFrame(disp_info, decoder_, stream_);
        // conversion to usable format, RGB, HxW, etc...
        // Output cleared, allow next decoding
        permits_[disp_info.CurrPicIdx].Push(1);
    }
}

}  // namespace cuda
}  // namespace decord