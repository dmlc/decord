/*!
 *  Copyright (c) 2019 by Contributors
 * \file filter_graph.h
 * \brief FFmpeg Filter Graph Definition
 */

#ifndef DECORD_VIDEO_FFMPEG_FILTER_GRAPH_H_
#define DECORD_VIDEO_FFMPEG_FILTER_GRAPH_H_

#include "ffmpeg_common.h"

#include <string>
#include <atomic>

#include <dmlc/base.h>

namespace decord {
namespace ffmpeg {

class FFMPEGFilterGraph {
    public:
        FFMPEGFilterGraph(std::string filter_desc, AVCodecContext *dec_ctx);
        void Push(AVFrame *frame);
        bool Pop(AVFrame **frame);
        ~FFMPEGFilterGraph();
    private:
        void Init(std::string filter_desc, AVCodecContext *dec_ctx);
        AVFilterContext *buffersink_ctx_;
        AVFilterContext *buffersrc_ctx_;
        AVFilterGraphPtr filter_graph_;
        std::atomic<int> count_;

    DISALLOW_COPY_AND_ASSIGN(FFMPEGFilterGraph);
};  // FFMPEGFilterGraph

}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_VIDEO_FFMPEG_FILTER_GRAPH_H_