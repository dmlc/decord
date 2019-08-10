/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
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

/**
 * \brief FFMPEGFilterGraph for filtering operations
 *
 */
class FFMPEGFilterGraph {
    public:
        /**
         * \brief Construct a new FFMPEGFilterGraph object
         *
         * \param filter_desc String defining filter descriptions
         * \param dec_ctx Decoder context
         */
        FFMPEGFilterGraph(std::string filter_desc, AVCodecContext *dec_ctx);
        /**
         * \brief Push frame to be processed into filter graph
         *
         * \param frame Pointer to AVFrame
         */
        void Push(AVFrame *frame);
        /**
         * \brief Pop filtered frame from graph
         *
         * \param frame Pointer to pointer to AVFrame
         * \return true Success
         * \return false Failed
         */
        bool Pop(AVFrame **frame);
        /**
         * \brief Destroy the FFMPEGFilterGraph object
         *
         */
        ~FFMPEGFilterGraph();
    private:
        /**
         * \brief Initialize filter graph
         *
         * \param filter_desc String defining filter descriptions
         * \param dec_ctx Decoder context
         */
        void Init(std::string filter_desc, AVCodecContext *dec_ctx);
        /**
         * \brief Buffer sink context, the output side of graph
         *
         */
        AVFilterContext *buffersink_ctx_;
        /**
         * \brief Buffer src context, the input side of graph
         *
         */
        AVFilterContext *buffersrc_ctx_;
        /**
         * \brief Smart pointer to filter graph
         *
         */
        AVFilterGraphPtr filter_graph_;
        /**
         * \brief Size of buffered frames under processing
         *
         */
        std::atomic<int> count_;

    DISALLOW_COPY_AND_ASSIGN(FFMPEGFilterGraph);
};  // FFMPEGFilterGraph

}  // namespace ffmpeg
}  // namespace decord
#endif  // DECORD_VIDEO_FFMPEG_FILTER_GRAPH_H_
