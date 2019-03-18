/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg_common-inl.h
 * \brief FFmpeg commons
 */

#ifndef DECORD_VIDEO_FFMPEG_COMMON_H_
#define DECORD_VIDEO_FFMPEG_COMMON_H_

#include <memory>
#include <queue>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/opt.h>
#ifdef __cplusplus
}
#endif

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {
// Deleter adaptor for functions like av_free that take a pointer.
template<typename T, typename R, R(*Fn)(T*)> struct Deleter {
    inline void operator() (T* p) const {
        if (p) Fn(p);
    }
};

// Deleter adaptor for functions like av_freep that take a pointer to a pointer.
template<typename T, typename R, R(*Fn)(T**)> struct Deleterp {
    inline void operator() (T* p) const {
        if (p) Fn(&p);
    }
};

// RAII adapter for raw FFMPEG structs

using AVFramePtr = std::shared_ptr<AVFrame>;

class AVFramePool {
    static const uint16_t MAX_AVFRAME_POOL_SIZE = 32;
    public:
        AVFramePool() : active_(true) {};
        ~AVFramePool() {
            LOG(INFO) << "Destructing AVFRAME Pool";
            active_ = false;
        }
        static AVFramePtr Alloc() {
            std::queue<AVFrame*> pool = Get()->pool_;
            if (pool.empty()) return AllocAVFrameWithDeleter();
            AVFrame* ret = pool.front();
            pool.pop();
            return std::shared_ptr<AVFrame>(ret, Recycle);
        }

        static AVFramePool* Get() {
            static AVFramePool pool;
            return &pool;
        }
    private:
        std::queue<AVFrame*> pool_;
        bool active_;

        static void Recycle(AVFrame *p) {
            if (!p) return;
            std::queue<AVFrame*> pool = Get()->pool_;
            if (!Get()->active_ || pool.size() + 1 > MAX_AVFRAME_POOL_SIZE) {
                av_frame_unref(p);
            } else {
                pool.push(p);
            }
        }

        static AVFramePtr AllocAVFrameWithDeleter() {
            return std::shared_ptr<AVFrame>(av_frame_alloc(), Recycle);
        }
};

using AVPacketPtr = std::shared_ptr<AVPacket>;

// inline void AVPacketDeleter(AVPacket *p) {
//     if (p) av_packet_unref(p);
// }

// inline AVPacketPtr AllocAVPacketWithDeleter() {
//     return std::shared_ptr<AVPacket>(av_packet_alloc(), AVPacketDeleter);
// }

class AVPacketPool {
    static const uint16_t MAX_AVPACKET_POOL_SIZE = 32;
    public:
        AVPacketPool() : active_(true) {};
        ~AVPacketPool() {
            active_ = false;
        }
        static AVPacketPtr Alloc() {
            std::queue<AVPacket*> pool = Get()->pool_;
            if (pool.empty()) return AllocAVPacketWithDeleter();
            AVPacket* ret = pool.front();
            pool.pop();
            return std::shared_ptr<AVPacket>(ret, Recycle);
        }

        static AVPacketPool* Get() {
            static AVPacketPool pool;
            return &pool;
        }
    private:
        std::queue<AVPacket*> pool_;
        bool active_;

        static void Recycle(AVPacket *p) {
            if (!p) return;
            std::queue<AVPacket*> pool = Get()->pool_;
            if (!Get()->active_ || pool.size() + 1 > MAX_AVPACKET_POOL_SIZE) {
                av_packet_unref(p);
            } else {
                pool.push(p);
            }
        }

        static AVPacketPtr AllocAVPacketWithDeleter() {
            return std::shared_ptr<AVPacket>(av_packet_alloc(), Recycle);
        }
};

using AVFormatContextPtr = std::unique_ptr<
    AVFormatContext, Deleter<AVFormatContext, void, avformat_free_context> >;

using AVCodecContextPtr = std::unique_ptr<
    AVCodecContext, Deleter<AVCodecContext, int, avcodec_close> >;\

using AVFilterGraphPtr = std::unique_ptr<
    AVFilterGraph, Deleterp<AVFilterGraph, void, avfilter_graph_free> >;

using AVFilterContextPtr = std::unique_ptr<
    AVFilterContext, Deleter<AVFilterContext, void, avfilter_free> >;

}  // namespace ffmpeg
}  // namespace decord



#endif  // DECORD_VIDEO_FFMPEG_COMMON_H_