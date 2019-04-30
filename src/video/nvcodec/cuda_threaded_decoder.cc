/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.cc
 * \brief NVCUVID based decoder impl
 */

#include "cuda_threaded_decoder.h"


namespace decord {
namespace cuda {

CUThreadedDecoder::CUThreadedDecoder(int device_id) 
    : device_id_(device_id), device_{}, ctx_{}, parser_{}, decoder_{}, 
    occupied_frames_(32), // 32 is cuvid's max number of decode surfaces
    pkt_queue_{}, buffer_queue_{}, frame_queue_{}, run_(false), frame_count_(0),
    tex_registry_() {
    
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

    ctx_ = CUContext(device_);
    if (!ctx_.initialized()) {
        LOG(FATAL) << "Problem initializing context";
        return;
    }

    parser_ = CUVideoParser(dec_ctx->codec, this, 20, dec_ctx->codecpar->extradata,
                            dec_ctx->codecpar->extradata_size);
    if (!parser_.Initialized()) {
        LOG(FATAL) << "Problem creating video parser";
        return;
    }
}

void CUThreadedDecoder::SetCodecContext(const AVCodecContext *dec_ctx, int width, int height) {
    CHECK(dec_ctx);
    width_ = width;
    height_ = height;
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    if (running) {
        Start();
    }
}

void CUThreadedDecoder::Start() {
    if (run_.load()) return;

    pkt_queue_.reset(new PacketQueue());
    frame_queue_.reset(new FrameQueue());
    avcodec_flush_buffers(dec_ctx_.get());
    CHECK(permits_.size() == 0);
    permits_.resize(kMaxOutputSurfaces);
    for (auto& p : permits_) {
        p.Push(1);
    }
    run_.store(true);
    // launch worker threads
    launcher_t_ = std::thread{&CUThreadedDecoder::LaunchThread, this};
    converter_t_ = std::thread{&CUThreadedDecoder::ConvertThread, this};
}

void CUThreadedDecoder::Stop() {
    if (run_.load()) {
        pkt_queue_->SignalForKill();
        run_.store(false);
        frame_queue_->SignalForKill();
        buffer_queue_->SignalForKill();
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
    surface_order_.clear();
    for (auto& p : permits_) {
        p.SignalForKill();
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
    frame_base_ = {static_cast<int>(format->frame_rate.denominator),
                   static_cast<int>(format->frame_rate.numerator)};
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
        return 0;
    }
    return 1;
}

int CUThreadedDecoder::HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info) {
    // push to converter
    buffer_queue_.Push(disp_info);
    // finished, send clear msg to allow next decoding
}

void CUThreadedDecoder::Push(AVPacketPtr pkt, DLTensor buf) {
    CHECK(run_.load());
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }
    pkt_queue_.Push(pkt);
    NumberedFrame nf = {};
    nf.t = buf;
    nf.n = -1;
    frame_queue_.Push(nf);  // push memory buffer
    ++frame_count_;
}

bool CUThreadedDecoder::Pop(DLTensor *frame) {
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    CHECK(surface_order_.size() > 0);
    int frame_num = surface_order_.pop_front();
    auto r = reorder_buffer_.find(frame_num);
    if (r == reorder_buffer_.end()) {
        return false;
    }
    *frame = (*r).t;
    reorder_buffer_.erase(r);
    --frame_count;
    return true;
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

        // calculate frame number for output order
        auto frame_num = av_rescale_q(pkt->pts, AV_TIME_BASE_Q, dec_ctx_->time_base);
        surface_order_.push(frame_num);
    }
}

void CUThreadedDecoder::ConvertThread() {
    context_.push();
    while (run_.load()) {
        bool ret;
        CUVIDPARSERDISPINFO *disp_info = nullptr;
        NumberedFrame nf;
        ret = buffer_queue_.Pop(&disp_info);
        if (!ret) return;
        CHECK(disp_info != nullptr);
        // CUDA mem buffer
        ret = frame_queue_.Pop(&nf);
        CHECK(nf.t.data != nullptr);
        uint8_t* dst = static_cast<uint8_t>(nf.t.data);
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
        ProcessFrame(textures.chroma, textures.luma, dst, stream_, input_width, input_height, width_, height_);
        int frame_num = av_rescale_q(frame.disp_info->timestamp,
                                      nv_time_base_, frame_base_);
        nf.n = frame_num;
        reorder_buffer_[frame_num] = nf;

        // Output cleared, allow next decoding
        permits_[disp_info.CurrPicIdx].Push(1);
    }
}

}  // namespace cuda
}  // namespace decord