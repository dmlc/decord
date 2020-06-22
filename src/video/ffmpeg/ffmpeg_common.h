/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file ffmpeg_common-inl.h
 * \brief FFmpeg commons
 */

#ifndef DECORD_VIDEO_FFMPEG_COMMON_H_
#define DECORD_VIDEO_FFMPEG_COMMON_H_

#include "../storage_pool.h"
#include <decord/base.h>
#include <decord/runtime/ndarray.h>
#include <decord/runtime/device_api.h>

#include <memory>
#include <queue>
#include <functional>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/opt.h>
#include <libavutil/version.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/file.h>
#ifdef DECORD_USE_LIBAVDEVICE
#include <libavdevice/avdevice.h>
#endif
#ifdef __cplusplus
}
#endif

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <dlpack/dlpack.h>

namespace decord {
namespace ffmpeg {
using NDArray = runtime::NDArray;

/**
 * \brief Deleter adaptor for functions like av_free that take a pointer.
 *
 * \tparam T Pointer type
 * \tparam R Deleter return type
 * \tparam R(*Fn)(T*) Real deleter function
 */
template<typename T, typename R, R(*Fn)(T*)> struct Deleter {
    inline void operator() (T* p) const {
        if (p) Fn(p);
    }
};

/**
 * \brief Deleter adaptor for functions like av_freep that take a pointer to a pointer.
 *
 * \tparam T Pointer type
 * \tparam R Deleter return type
 * \tparam R(*Fn)(T*) Real deteter function
 */
template<typename T, typename R, R(*Fn)(T**)> struct Deleterp {
    inline void operator() (T* p) const {
        if (p) Fn(&p);
    }
};

/**
 * \brief AutoReleasePool for AVFrame
 *
 * \tparam S Pool size
 */
template<int S>
class AutoReleaseAVFramePool : public AutoReleasePool<AVFrame, S> {
    public:
        static AutoReleaseAVFramePool<S>* Get() {
            static AutoReleaseAVFramePool<S> pool;
            return &pool;
        }

    private:
        AVFrame* Allocate() final {
            return av_frame_alloc();
        }

        void Delete(AVFrame* p) final {
            av_frame_free(&p);
        }
};

/**
 * \brief AutoReleasePool for AVPacket
 *
 * \tparam S Pool size
 */
template<int S>
class AutoReleaseAVPacketPool : public AutoReleasePool<AVPacket, S> {
    public:
        static AutoReleaseAVPacketPool<S>* Get() {
            static AutoReleaseAVPacketPool<S> pool;
            return &pool;
        }

    private:
        AVPacket* Allocate() final {
            return av_packet_alloc();
        }

        void Delete(AVPacket* p) final {
            av_packet_free(&p);
        }
};

// RAII adapter for raw FFMPEG structs

/**
 * \brief maximum pool size for AVFrame, per thread
 *
 */
static const int kAVFramePoolMaxSize = 0;
/**
 * \brief maximum pool size for AVPacket, per thread
 *
 */
static const int kAVPacketPoolMaxSize = 0;
/**
 * \brief AVFramePool
 *
 */
using AVFramePool = AutoReleaseAVFramePool<kAVFramePoolMaxSize>;
/**
 * \brief AVPacketPool
 *
 */
using AVPacketPool = AutoReleaseAVPacketPool<kAVPacketPoolMaxSize>;
/**
 * \brief Smart pointer for AVFrame
 *
 */
using AVFramePtr = std::shared_ptr<AVFrame>;
/**
 * \brief Smart pointer for AVPacket
 *
 */
using AVPacketPtr = std::shared_ptr<AVPacket>;

/**
 * \brief Smart pointer for AVFormatContext, non copyable
 *
 */
using AVFormatContextPtr = std::unique_ptr<
    AVFormatContext, Deleterp<AVFormatContext, void, avformat_close_input> >;

/**
 * \brief Smart pointer for AVCodecContext, non copyable
 *
 */
using AVCodecContextPtr = std::unique_ptr<
    AVCodecContext, Deleterp<AVCodecContext, void, avcodec_free_context> >;

/**
 * \brief Smart pointer for AVFilterGraph, non copyable
 *
 */
using AVFilterGraphPtr = std::unique_ptr<
    AVFilterGraph, Deleterp<AVFilterGraph, void, avfilter_graph_free> >;

/**
 * \brief Smart pointer for AVFilterContext, non copyable
 *
 */
using AVFilterContextPtr = std::unique_ptr<
    AVFilterContext, Deleter<AVFilterContext, void, avfilter_free> >;


using AVBSFContextPtr = std::unique_ptr<
    AVBSFContext, Deleterp<AVBSFContext, void, av_bsf_free> >;

using AVCodecParametersPtr = std::unique_ptr<
    AVCodecParameters, Deleterp<AVCodecParameters, void, avcodec_parameters_free> >;


inline void ToDLTensor(AVFramePtr p, DLTensor& dlt, int64_t *shape) {
	CHECK(p) << "Error: converting empty AVFrame to DLTensor";
	// int channel = p->linesize[0] / p->width;
	CHECK(AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 || AVPixelFormat(p->format) == AV_PIX_FMT_GRAY8)
        << "Only support RGB24/GRAY8 image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    CHECK(p->linesize[0] % p->width == 0)
        << "AVFrame data is not a compact array. linesize: " << p->linesize[0]
        << " width: " << p->width;

	DLContext ctx;
	if (p->hw_frames_ctx) {
        LOG(FATAL) << "HW ctx not supported";
		ctx = DLContext({ kDLGPU, 0 });
	}
	else {
		ctx = kCPU;
	}
	// LOG(INFO) << p->height << " x";
	// std::vector<int64_t> shape = { p->height, p->width, p->linesize[0] / p->width };
    // LOG(INFO) << p->height << " x " << p->width;
	shape[0] = p->height;
	shape[1] = p->width;
	shape[2] = p->linesize[0] / p->width;
	dlt.data = p->data[0];
	dlt.ctx = ctx;
	dlt.ndim = 3;
	dlt.dtype = kUInt8;
	dlt.shape = shape;
    dlt.strides = NULL;
	dlt.byte_offset = 0;
}

struct AVFrameManager {
	AVFramePtr ptr;
  int64_t shape[3];
	explicit AVFrameManager(AVFramePtr p) : ptr(p) {}
};

struct AVIOBufferData {
    uint8_t *ptr;
    size_t size;  ///< size left in the buffer
    uint8_t *ori_ptr;
    size_t file_size;
};

class AVIOBytesContext {
  public:
    AVIOBytesContext() {}

    AVIOBytesContext(std::string data, size_t buffer_size): ctx_(nullptr), data_(data) {
        auto buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
        if (!buffer) {
            LOG(WARNING) << "Unable to allocate AVIOBytes buffer with size: " << buffer_size;
            return;
        }
        bd_.ptr = reinterpret_cast<uint8_t*>(&data_[0]);
        bd_.ori_ptr = bd_.ptr;
        bd_.size = data_.size();
        bd_.file_size = bd_.size;

        ctx_ = avio_alloc_context(buffer, buffer_size, 0, &bd_, 
                                  &AVIOBytesContext::read, 0, &AVIOBytesContext::seek);
        if (!ctx_) {
            LOG(WARNING) << "Unable to allocate AVIOContext!";
            return;
        }
        
    }

    ~AVIOBytesContext() {
        if (ctx_) {
            av_freep(&ctx_->buffer);
        }
        avio_context_free(&ctx_);
    }

    AVIOContext* get_avio() {
        return ctx_;
    }

    static int read(void *opaque, uint8_t *buf, int buf_size) {
        struct AVIOBufferData *bd = (struct AVIOBufferData *)opaque;
        buf_size = FFMIN(buf_size, bd->size);
        if (!buf_size)
            return AVERROR_EOF;
        /* copy internal buffer data to buf */
        memcpy(buf, bd->ptr, buf_size);
        bd->ptr  += buf_size;
        bd->size -= buf_size;
        return buf_size;
    }

    static int64_t seek(void *opaque, int64_t offset, int whence) {
        AVIOBufferData *bd = (AVIOBufferData *)opaque;
        int64_t ret = -1;

        switch (whence)
        {
            case AVSEEK_SIZE:
                ret = bd->file_size;
                break;
            case SEEK_SET:
                bd->ptr = bd->ori_ptr + offset;
                bd->size = bd->file_size - offset;
                ret = offset;
                break;
            default:
                break;
        }
        return ret;
    }

  private:
    AVIOContext* ctx_;
    struct AVIOBufferData bd_;
    std::string data_;
};

}  // namespace ffmpeg
}  // namespace decord
#endif  // DECORD_VIDEO_FFMPEG_COMMON_H_
