/*!
 *  Copyright (c) 2024 by Contributors if not otherwise specified
 * \file videotoolbox_threaded_decoder.cc
 * \brief VideoToolbox based decoder implementation for macOS GPU acceleration
 */

#include "videotoolbox_threaded_decoder.h"

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#ifdef __APPLE__
#include <VideoToolbox/VideoToolbox.h>
#include <CoreVideo/CoreVideo.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#endif

namespace decord {
namespace videotoolbox {

VideoToolboxThreadedDecoder::VideoToolboxThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat)
    : device_id_(device_id)
    , run_(false)
    , frame_count_(0)
    , draining_(false)
    , initialized_(false)
    , width_(0)
    , height_(0)
#ifdef __APPLE__
    , decompression_session_(nullptr)
    , format_description_(nullptr)
#endif
    , error_status_(false) {

    pkt_queue_ = std::unique_ptr<PacketQueue>(new PacketQueue());
    frame_queue_ = std::unique_ptr<FrameQueue>(new FrameQueue());

    InitBitStreamFilter(codecpar, iformat);

    // Setup VideoToolbox decoder
    if (!SetupVideoToolboxDecoder(codecpar)) {
        LOG(FATAL) << "Failed to setup VideoToolbox decoder for device " << device_id_;
    }
}

VideoToolboxThreadedDecoder::~VideoToolboxThreadedDecoder() {
    Stop();
    CleanupVideoToolboxDecoder();
}

void VideoToolboxThreadedDecoder::InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat) {
#ifdef __APPLE__
    const AVBitStreamFilter *bsf = nullptr;

    // Select appropriate bitstream filter based on codec
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
            break;
        case AV_CODEC_ID_HEVC:
            bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            break;
        case AV_CODEC_ID_AV1:
            // AV1 doesn't typically need bitstream filtering for VideoToolbox
            // The raw AV1 stream should work directly
            LOG(INFO) << "AV1 codec detected, using raw stream (no bitstream filter needed)";
            return;
        case AV_CODEC_ID_VP9:
            // VP9 doesn't typically need bitstream filtering for VideoToolbox
            // The raw VP9 stream should work directly
            LOG(INFO) << "VP9 codec detected, using raw stream (no bitstream filter needed)";
            return;
        case AV_CODEC_ID_PRORES:
        case AV_CODEC_ID_PRORES_RAW:
            // ProRes doesn't need bitstream filtering
            LOG(INFO) << "ProRes codec detected, using raw stream (no bitstream filter needed)";
            return;
        default:
            LOG(WARNING) << "No bitstream filter available for codec: " << codecpar->codec_id;
            return;
    }

    if (!bsf) {
        LOG(WARNING) << "Bitstream filter not found";
        return;
    }

    AVBSFContext *bsf_ctx = nullptr;
    CHECK_GE(av_bsf_alloc(bsf, &bsf_ctx), 0) << "Failed to allocate bitstream filter";
    bsf_ctx_ = std::unique_ptr<AVBSFContext, ffmpeg::Deleterp<AVBSFContext, void, av_bsf_free>>(bsf_ctx);
    CHECK_GE(avcodec_parameters_copy(bsf_ctx_->par_in, codecpar), 0) << "Failed to copy codec parameters to BSF";
    CHECK_GE(av_bsf_init(bsf_ctx_.get()), 0) << "Failed to initialize bitstream filter";
#endif
}

bool VideoToolboxThreadedDecoder::SetupVideoToolboxDecoder(AVCodecParameters *codecpar) {
#ifdef __APPLE__
    OSStatus status;

    // Create format description from codec parameters
    CMVideoFormatDescriptionRef format_desc = nullptr;

    // Create extradata dictionary
    CFMutableDictionaryRef extensions = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    if (codecpar->extradata && codecpar->extradata_size > 0) {
        CFDataRef extradata = CFDataCreate(kCFAllocatorDefault, codecpar->extradata, codecpar->extradata_size);
        CFDictionarySetValue(extensions, CFSTR("SampleDescriptionExtensionAtoms"), extradata);
        CFRelease(extradata);
    }

    // Create format description based on codec type
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                   kCMVideoCodecType_H264,
                                                   codecpar->width,
                                                   codecpar->height,
                                                   extensions,
                                                   &format_desc);
            break;
        case AV_CODEC_ID_HEVC:
            status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                   kCMVideoCodecType_HEVC,
                                                   codecpar->width,
                                                   codecpar->height,
                                                   extensions,
                                                   &format_desc);
            break;
        case AV_CODEC_ID_PRORES:
            // ProRes codec - detect the specific variant from codec parameters
            {
                CMVideoCodecType prores_type = DetectProResVariant(codecpar);
                status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                       prores_type,
                                                       codecpar->width,
                                                       codecpar->height,
                                                       extensions,
                                                       &format_desc);
            }
            break;
        case AV_CODEC_ID_PRORES_RAW:
            status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                   kCMVideoCodecType_AppleProResRAW,
                                                   codecpar->width,
                                                   codecpar->height,
                                                   extensions,
                                                   &format_desc);
            break;
        case AV_CODEC_ID_AV1:
            status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                   kCMVideoCodecType_AV1,
                                                   codecpar->width,
                                                   codecpar->height,
                                                   extensions,
                                                   &format_desc);
            break;
        case AV_CODEC_ID_VP9:
            status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                                   kCMVideoCodecType_VP9,
                                                   codecpar->width,
                                                   codecpar->height,
                                                   extensions,
                                                   &format_desc);
            break;
        default:
            LOG(ERROR) << "Unsupported codec for VideoToolbox: " << codecpar->codec_id;
            CFRelease(extensions);
            return false;
    }

    CFRelease(extensions);

    if (status != noErr) {
        LOG(ERROR) << "Failed to create format description: " << status;
        return false;
    }

    format_description_ = format_desc;

    // Create decompression session
    VTDecompressionOutputCallbackRecord callback_record = {
        VideoToolboxThreadedDecoder::VTDecompressionOutputCallback,
        this
    };

    // Create session attributes
    CFMutableDictionaryRef session_attrs = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    // Enable hardware acceleration
    CFDictionarySetValue(session_attrs, kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder, kCFBooleanTrue);

    // Create output attributes
    CFMutableDictionaryRef output_attrs = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    // Request BGRA pixel format for easier conversion
    int32_t pixel_format_value = kCVPixelFormatType_32BGRA;
    CFNumberRef pixel_format = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel_format_value);
    CFDictionarySetValue(output_attrs, kCVPixelBufferPixelFormatTypeKey, pixel_format);
    CFRelease(pixel_format);

    status = VTDecompressionSessionCreate(
        kCFAllocatorDefault,
        format_description_,
        session_attrs,
        output_attrs,
        &callback_record,
        &decompression_session_);

    CFRelease(session_attrs);
    CFRelease(output_attrs);

    if (status != noErr) {
        LOG(ERROR) << "Failed to create decompression session: " << status;
        return false;
    }

    initialized_ = true;
    LOG(INFO) << "VideoToolbox decoder initialized successfully";
    return true;
#else
    LOG(ERROR) << "VideoToolbox is only available on macOS";
    return false;
#endif
}

void VideoToolboxThreadedDecoder::CleanupVideoToolboxDecoder() {
#ifdef __APPLE__
    if (decompression_session_) {
        VTDecompressionSessionInvalidate(decompression_session_);
        CFRelease(decompression_session_);
        decompression_session_ = nullptr;
    }

    if (format_description_) {
        CFRelease(format_description_);
        format_description_ = nullptr;
    }

    initialized_ = false;
#endif
}

void VideoToolboxThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation) {
    // For VideoToolbox, we don't need to copy the context as we use our own decoder
    dec_ctx_ = std::unique_ptr<AVCodecContext, ffmpeg::Deleterp<AVCodecContext, void, avcodec_free_context>>(avcodec_alloc_context3(nullptr));

    width_ = width > 0 ? width : dec_ctx->width;
    height_ = height > 0 ? height : dec_ctx->height;

    // Set time base
    vt_time_base_ = dec_ctx->time_base;
    frame_base_ = dec_ctx->framerate;
}

bool VideoToolboxThreadedDecoder::Initialized() const {
    return initialized_.load();
}

void VideoToolboxThreadedDecoder::Start() {
    if (run_.load()) return;

    run_ = true;
    draining_ = false;
    frame_count_ = 0;

    launcher_t_ = std::thread(&VideoToolboxThreadedDecoder::LaunchThread, this);
}

void VideoToolboxThreadedDecoder::Stop() {
    if (!run_.load()) return;

    run_ = false;
    draining_ = true;

    // Signal end of stream
    AVPacketPtr null_pkt(nullptr);
    pkt_queue_->Push(std::move(null_pkt));

    if (launcher_t_.joinable()) {
        launcher_t_.join();
    }
}

void VideoToolboxThreadedDecoder::Clear() {
    // Clear queues
    AVPacketPtr pkt;
    while (pkt_queue_->Pop(&pkt)) {
        // Just drain the queue
    }

    NDArray frame;
    while (frame_queue_->Pop(&frame)) {
        // Just drain the queue
    }

    // Clear frame buffer
    {
        std::lock_guard<std::mutex> lock(frame_buffer_mutex_);
        frame_buffer_.clear();
    }

    frame_count_ = 0;
}

void VideoToolboxThreadedDecoder::Push(AVPacketPtr pkt, NDArray buf) {
    pkt_queue_->Push(std::move(pkt));
}

bool VideoToolboxThreadedDecoder::Pop(NDArray *frame) {
    return frame_queue_->Pop(frame);
}

void VideoToolboxThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    for (auto d : dts) {
        discard_pts_.insert(d);
    }
}

void VideoToolboxThreadedDecoder::ClearDiscardPTS() {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.clear();
}

void VideoToolboxThreadedDecoder::LaunchThread() {
    LaunchThreadImpl();
}

void VideoToolboxThreadedDecoder::LaunchThreadImpl() {
    while (run_.load()) {
        AVPacketPtr pkt;
        if (!pkt_queue_->Pop(&pkt)) {
            break;
        }

        if (!pkt) {
            // End of stream
            draining_ = true;
            break;
        }

        // Check if we should discard this packet
        {
            std::lock_guard<std::mutex> lock(pts_mutex_);
            if (discard_pts_.find(pkt->pts) != discard_pts_.end()) {
                continue;
            }
        }

#ifdef __APPLE__
        // Apply bitstream filter if available
        AVPacketPtr filtered_pkt = ffmpeg::AVPacketPool::Get()->Acquire();
        if (filtered_pkt->data) {
            av_packet_unref(filtered_pkt.get());
        }

        if (bsf_ctx_) {
            CHECK_GE(av_bsf_send_packet(bsf_ctx_.get(), pkt.get()), 0) << "Error sending BSF packet";
            int bsf_ret;
            while ((bsf_ret = av_bsf_receive_packet(bsf_ctx_.get(), filtered_pkt.get())) == 0) {
                // Decode the filtered packet
                DecodePacket(filtered_pkt.get());
            }
        } else {
            // Decode packet directly
            DecodePacket(pkt.get());
        }
#endif
    }
}

#ifdef __APPLE__
void VideoToolboxThreadedDecoder::DecodePacket(AVPacket *pkt) {
    if (!decompression_session_ || !pkt->data) {
        return;
    }

    // Create CMSampleBuffer from AVPacket
    CMBlockBufferRef block_buffer = nullptr;
    OSStatus status = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault,
        pkt->data,
        pkt->size,
        kCFAllocatorNull,
        nullptr,
        0,
        pkt->size,
        0,
        &block_buffer);

    if (status != noErr) {
        LOG(ERROR) << "Failed to create block buffer: " << status;
        return;
    }

    CMSampleBufferRef sample_buffer = nullptr;
    size_t sample_size = pkt->size;
    status = CMSampleBufferCreateReady(
        kCFAllocatorDefault,
        block_buffer,
        format_description_,
        1,
        0,
        nullptr,
        1,
        &sample_size,
        &sample_buffer);

    CFRelease(block_buffer);

    if (status != noErr) {
        LOG(ERROR) << "Failed to create sample buffer: " << status;
        return;
    }

    // Set presentation timestamp
    CMTime pts = CMTimeMake(pkt->pts, vt_time_base_.den);
    CMSampleBufferSetOutputPresentationTimeStamp(sample_buffer, pts);

    // Decode the frame
    VTDecodeInfoFlags info_flags = 0;
    status = VTDecompressionSessionDecodeFrame(
        decompression_session_,
        sample_buffer,
        kVTDecodeFrame_EnableAsynchronousDecompression,
        sample_buffer,
        &info_flags);

    CFRelease(sample_buffer);

    if (status != noErr) {
        LOG(ERROR) << "Failed to decode frame: " << status;
    }
}
#endif

runtime::NDArray VideoToolboxThreadedDecoder::ConvertCVImageBufferToNDArray(CVImageBufferRef imageBuffer) {
#ifdef __APPLE__
    // Lock the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    size_t bytes_per_row = CVPixelBufferGetBytesPerRow(imageBuffer);

    void *base_address = CVPixelBufferGetBaseAddress(imageBuffer);
    OSType pixel_format = CVPixelBufferGetPixelFormatType(imageBuffer);

    // Create NDArray
    std::vector<int64_t> shape = {static_cast<int64_t>(height), static_cast<int64_t>(width), 3};
    DLContext ctx = kCPU; // We'll copy to CPU for now
    DLDataType dtype = kUInt8;

    NDArray ndarray = NDArray::Empty(shape, dtype, ctx);

    // Copy data based on pixel format
    if (pixel_format == kCVPixelFormatType_32BGRA) {
        // Convert BGRA to RGB
        uint8_t *src = static_cast<uint8_t*>(base_address);
        uint8_t *dst = static_cast<uint8_t*>(ndarray->data);

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t src_idx = y * bytes_per_row + x * 4;
                size_t dst_idx = (y * width + x) * 3;

                // BGRA to RGB
                dst[dst_idx + 0] = src[src_idx + 2]; // R
                dst[dst_idx + 1] = src[src_idx + 1]; // G
                dst[dst_idx + 2] = src[src_idx + 0]; // B
            }
        }
    } else {
        LOG(WARNING) << "Unsupported pixel format: " << pixel_format;
        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        return runtime::NDArray();
    }

    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    return ndarray;
#else
    return runtime::NDArray();
#endif
}

void VideoToolboxThreadedDecoder::VTDecompressionOutputCallback(
    void *decompressionOutputRefCon,
    void *sourceFrameRefCon,
    OSStatus status,
    VTDecodeInfoFlags infoFlags,
    CVImageBufferRef imageBuffer,
    CMTime presentationTimeStamp,
    CMTime presentationDuration) {

    VideoToolboxThreadedDecoder *decoder = static_cast<VideoToolboxThreadedDecoder*>(decompressionOutputRefCon);

    if (status != noErr) {
        LOG(ERROR) << "VideoToolbox decode error: " << status;
        return;
    }

    if (!imageBuffer) {
        return;
    }

    // Convert CVImageBuffer to NDArray
    NDArray frame = decoder->ConvertCVImageBufferToNDArray(imageBuffer);

    if (frame.defined()) {
        decoder->frame_queue_->Push(std::move(frame));
        decoder->frame_count_++;
    }
}

void VideoToolboxThreadedDecoder::RecordInternalError(std::string message) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_message_ = message;
    error_status_ = true;
}

void VideoToolboxThreadedDecoder::CheckErrorStatus() {
    if (error_status_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        LOG(FATAL) << error_message_;
    }
}

#ifdef __APPLE__
CMVideoCodecType VideoToolboxThreadedDecoder::DetectProResVariant(AVCodecParameters *codecpar) {
    // Default to ProRes 422
    CMVideoCodecType prores_type = kCMVideoCodecType_AppleProRes422;

    // Try to detect ProRes variant from codec name or profile
    if (codecpar->profile != AV_PROFILE_UNKNOWN) {
        switch (codecpar->profile) {
            case AV_PROFILE_PRORES_4444:
                prores_type = kCMVideoCodecType_AppleProRes4444;
                break;
            case AV_PROFILE_PRORES_XQ:
                prores_type = kCMVideoCodecType_AppleProRes4444XQ;
                break;
            case AV_PROFILE_PRORES_HQ:
                prores_type = kCMVideoCodecType_AppleProRes422HQ;
                break;
            case AV_PROFILE_PRORES_STANDARD:
                prores_type = kCMVideoCodecType_AppleProRes422;
                break;
            case AV_PROFILE_PRORES_LT:
                prores_type = kCMVideoCodecType_AppleProRes422LT;
                break;
            case AV_PROFILE_PRORES_PROXY:
                prores_type = kCMVideoCodecType_AppleProRes422Proxy;
                break;
            default:
                // Unknown profile, use default
                LOG(INFO) << "Unknown ProRes profile: " << codecpar->profile << ", using default ProRes 422";
                break;
        }
    }

    // Additional detection based on bit depth and chroma format
    if (codecpar->bits_per_coded_sample > 8) {
        // High bit depth suggests 4444 variant
        if (prores_type == kCMVideoCodecType_AppleProRes422) {
            prores_type = kCMVideoCodecType_AppleProRes422HQ;
        }
    }

    LOG(INFO) << "Detected ProRes variant: " << prores_type;
    return prores_type;
}
#endif

}  // namespace videotoolbox
}  // namespace decord