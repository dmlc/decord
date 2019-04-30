/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file ffmpeg_common-inl.h
 * \brief FFmpeg commons
 */

#ifndef DECORD_VIDEO_FFMPEG_COMMON_H_
#define DECORD_VIDEO_FFMPEG_COMMON_H_

#include <memory>
#include <queue>
#include <functional>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/opt.h>
#include <libavutil/version.h>
#ifdef __cplusplus
}
#endif

#include "../storage_pool.h"

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

namespace decord {
namespace ffmpeg {
/**
 * \brief Deleter adaptor for functions like av_free that take a pointer.
 * 
 * \tparam T Pointer type
 * \tparam R Deleter return type
 * \tparam R(*Fn)(T*) Real deteter function
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
            av_frame_unref(p);
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
            av_packet_unref(p);
        }
};

// RAII adapter for raw FFMPEG structs

/**
 * \brief maximum pool size for AVFrame, per thread
 * 
 */
static const int kAVFramePoolMaxSize = 32;
/**
 * \brief maximum pool size for AVPacket, per thread
 * 
 */
static const int kAVPacketPoolMaxSize = 32;
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
    AVFormatContext, Deleter<AVFormatContext, void, avformat_free_context> >;

/**
 * \brief Smart pointer for AVCodecContext, non copyable
 * 
 */
using AVCodecContextPtr = std::unique_ptr<
    AVCodecContext, Deleter<AVCodecContext, int, avcodec_close> >;

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

}  // namespace ffmpeg
}  // namespace decord
#endif  // DECORD_VIDEO_FFMPEG_COMMON_H_