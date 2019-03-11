/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg_common-inl.h
 * \brief FFmpeg commons
 */

#ifndef DECORD_VIDEO_FFMPEG_COMMON_H_
#define DECORD_VIDEO_FFMPEG_COMMON_H_

#include <memory>

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

void AVFrameDeleter(AVFrame *p) {
    if (p) av_frame_free(&p);
}

AVFramePtr AllocAVFrameWithDeleter() {
    return std::shared_ptr<AVFrame>(av_frame_alloc(), AVFrameDeleter);
}

using AVPacketPtr = std::shared_ptr<AVPacket>;

void AVPacketDeleter(AVPacket *p) {
    if (p) av_packet_free(&p);
}

AVPacketPtr AllocAVPacketWithDeleter() {
    return std::shared_ptr<AVPacket>(av_packet_alloc(), AVPacketDeleter);
}

using AVFormatContextPtr = std::unique_ptr<
    AVFormatContext, Deleter<AVFormatContext, void, avformat_free_context> >;

using AVCodecContextPtr = std::unique_ptr<
    AVCodecContext, Deleter<AVCodecContext, int, avcodec_close> >;

using AVFilterGraphPtr = std::unique_ptr<
    AVFilterGraph, Deleterp<AVFilterGraph, void, avfilter_graph_free> >;

using AVFilterContextPtr = std::unique_ptr<
    AVFilterContext, Deleter<AVFilterContext, void, avfilter_free> >;

}  // namespace ffmpeg
}  // namespace decord



#endif  // DECORD_VIDEO_FFMPEG_COMMON_H_