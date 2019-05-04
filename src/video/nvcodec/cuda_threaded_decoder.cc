/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.cc
 * \brief NVCUVID based decoder impl
 */

#include "cuda_threaded_decoder.h"
#include "cuda_mapped_frame.h"
#include "cuda_texture.h"
#include "../../improc/improc.h"
#include "nvcuvid/nvcuvid.h"
#include <nvml.h>


namespace decord {
namespace cuda {
using namespace runtime;

CUThreadedDecoder::CUThreadedDecoder(int device_id) 
    : device_id_(device_id), stream_({-1, false}), device_{}, ctx_{}, parser_{}, decoder_{}, 
    pkt_queue_{}, frame_queue_{}, buffer_queue_{}, reorder_buffer_{}, surface_order_{},
    permits_{}, run_(false), frame_count_(0), draining_(false),
    tex_registry_(), nv_time_base_({1, 10000000}), frame_base_({1, 1000000}),
    dec_ctx_(nullptr), width_(-1), height_(-1) {
    
    CHECK_CUDA_CALL(cuInit(0));
    CHECK_CUDA_CALL(cuDeviceGet(&device_, device_id_));

    char device_name[100];
    CHECK_CUDA_CALL(cuDeviceGetName(device_name, 100, device_));
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
            stream_ = CUStream(device_id_, true);
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
        stream_ = CUStream(device_id_, true);
    }

    ctx_ = CUContext(device_);
    if (!ctx_.Initialized()) {
        LOG(FATAL) << "Problem initializing context";
        return;
    }
}

void CUThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height) {
    CHECK(dec_ctx);
    LOG(INFO) << "SetCodecContext";
    width_ = width;
    height_ = height;
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    parser_ = CUVideoParser(dec_ctx->codec_id, this, kMaxOutputSurfaces, dec_ctx->extradata,
                            dec_ctx->extradata_size);
    if (!parser_.Initialized()) {
        LOG(FATAL) << "Problem creating video parser";
        return;
    }
    if (running) {
        Start();
    }
    LOG(INFO) << "Finish SetCOdecContext...";
}

void CUThreadedDecoder::Start() {
    if (run_.load()) return;

    LOG(INFO) << "Starting...";
    pkt_queue_.reset(new PacketQueue());
    frame_queue_.reset(new FrameQueue());
    LOG(INFO) << "Reset done.";
    avcodec_flush_buffers(dec_ctx_.get());
    CHECK(permits_.size() == 0);
    LOG(INFO) << "resizing permits";
    permits_.resize(kMaxOutputSurfaces);
    LOG(INFO) << "init permits";
    for (auto p : permits_) {
        p.reset(new PermitQueue());
        p->Push(1);
    }
    LOG(INFO) << "permits initied.";
    run_.store(true);
    // launch worker threads
    LOG(INFO) << "launching workers";
    launcher_t_ = std::thread{&CUThreadedDecoder::LaunchThread, this};
    // converter_t_ = std::thread{&CUThreadedDecoder::ConvertThread, this};
    LOG(INFO) << "finish launching workers";
}

void CUThreadedDecoder::Stop() {
    if (run_.load()) {
        pkt_queue_->SignalForKill();
        run_.store(false);
        frame_queue_->SignalForKill();
        buffer_queue_->SignalForKill();
        for (auto p : permits_) {
            if (p) p->SignalForKill();
        }
    }
    if (launcher_t_.joinable()) {
        launcher_t_.join();
    }
    if (converter_t_.joinable()) {
        converter_t_.join();
    }
}

void CUThreadedDecoder::Clear() {
    Stop();
    frame_count_.store(0);
    reorder_buffer_.clear();
    std::queue<int> tmp_queue;
    std::swap(tmp_queue, surface_order_);
    // surface_order_.clear();
    for (auto& p : permits_) {
        p->SignalForKill();
    }
    permits_.clear();
}

CUThreadedDecoder::~CUThreadedDecoder() {
    Clear();
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
    width_ = format->coded_width;
    height_ = format->coded_height;
    frame_base_ = {static_cast<int>(format->frame_rate.denominator),
                   static_cast<int>(format->frame_rate.numerator)};
    return decoder_.Initialize(format);
}

int CUThreadedDecoder::HandlePictureDecode_(CUVIDPICPARAMS* pic_params) {
    CHECK_GE(pic_params->CurrPicIdx, 0);
    CHECK_LT(pic_params->CurrPicIdx, permits_.size());
    auto permit_queue = permits_[pic_params->CurrPicIdx];
    int tmp;
    if (!run_.load() || !permit_queue->Pop(&tmp)) return 0;
    if (!CHECK_CUDA_CALL(cuvidDecodePicture(decoder_, pic_params))) {
        LOG(FATAL) << "Failed to launch cuvidDecodePicture";
        return 0;
    }
    return 1;
}

int CUThreadedDecoder::HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info) {
    // push to converter
    buffer_queue_->Push(disp_info);
    // finished, send clear msg to allow next decoding
    return 1;
}

void CUThreadedDecoder::Push(AVPacketPtr pkt, NDArray buf) {
    CHECK(run_.load());
    LOG(INFO) << "Pusing CUDecoder";
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }
    pkt_queue_->Push(pkt);
    frame_queue_->Push(buf);  // push memory buffer
    ++frame_count_;
}

bool CUThreadedDecoder::Pop(NDArray *frame) {
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    CHECK(surface_order_.size() > 0);
    int frame_num = surface_order_.front();
    surface_order_.pop();
    auto r = reorder_buffer_.find(frame_num);
    if (r == reorder_buffer_.end()) {
        return false;
    }
    *frame = (*r).second;
    reorder_buffer_.erase(r);
    --frame_count_;
    return true;
}

void CUThreadedDecoder::LaunchThread() {
    ctx_.Push();
    LOG(INFO) << "LaunchThread, pushed";
    while (run_.load()) {
        bool ret;
        AVPacketPtr avpkt = nullptr;
        LOG(INFO) << "LaunchThread, poping";
        ret = pkt_queue_->Pop(&avpkt);
        LOG(INFO) << "LaunchThread, poped";
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

        // calculate frame number for output order
        auto frame_num = av_rescale_q(avpkt->pts, AV_TIME_BASE_Q, dec_ctx_->time_base);
        surface_order_.push(frame_num);
    }
}

void CUThreadedDecoder::ConvertThread() {
    ctx_.Push();
    LOG(INFO) << "ConvertThrad, pushed";
    while (run_.load()) {
        bool ret;
        CUVIDPARSERDISPINFO *disp_info = nullptr;
        NDArray arr;
        LOG(INFO) << "ConvertThrad, poping";
        ret = buffer_queue_->Pop(&disp_info);
        LOG(INFO) << "ConvertThrad, poped";
        if (!ret) return;
        CHECK(disp_info != nullptr);
        // CUDA mem buffer
        ret = frame_queue_->Pop(&arr);
        CHECK(arr.defined());
        uint8_t* dst_ptr = static_cast<uint8_t*>(arr.data_->dl_tensor.data);
        auto frame = CUMappedFrame(disp_info, decoder_, stream_);
        // conversion to usable format, RGB, resize, etc...
        auto input_width = decoder_.Width();
        auto input_height = decoder_.Height();
        auto& textures = tex_registry_.GetTexture(frame.get_ptr(),
                                                  frame.get_pitch(),
                                                  input_width,
                                                  input_height,
                                                  ScaleMethod_Linear,
                                                  ChromaUpMethod_Linear);
        ProcessFrame(textures.chroma, textures.luma, dst_ptr, stream_, input_width, input_height, width_, height_);
        int frame_num = av_rescale_q(frame.disp_info->timestamp,
                                      nv_time_base_, frame_base_);
        reorder_buffer_[frame_num] = arr;

        // Output cleared, allow next decoding
        permits_[disp_info->picture_index]->Push(1);
    }
}

}  // namespace cuda
}  // namespace decord