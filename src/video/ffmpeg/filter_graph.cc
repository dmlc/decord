/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file filter_graph.cc
 * \brief FFmpeg Filter Graph Impl
 */

#include "filter_graph.h"

#include <dmlc/logging.h>
extern "C" {
#include <libavutil/pixdesc.h>
}

namespace decord {
namespace ffmpeg {

FFMPEGFilterGraph::FFMPEGFilterGraph(std::string filters_descr, AVCodecContext *dec_ctx)
    : buffersink_ctx_(nullptr), buffersrc_ctx_(nullptr), filter_graph_(nullptr), count_(0) {
    Init(filters_descr, dec_ctx);
}

FFMPEGFilterGraph::~FFMPEGFilterGraph() {
    // avfilter_free(buffersink_ctx_);
    // avfilter_free(buffersrc_ctx_);
    // avfilter_graph_free(&filter_graph_);
}

void FFMPEGFilterGraph::Init(std::string filters_descr, AVCodecContext *dec_ctx) {
    char args[512];
    #if LIBAVFILTER_VERSION_INT < AV_VERSION_INT(7,14,100)
    avfilter_register_all();
    #endif
    const AVFilter *buffersrc  = avfilter_get_by_name("buffer");
	const AVFilter *buffersink = avfilter_get_by_name("buffersink");
    if (!buffersink) {
        buffersink = avfilter_get_by_name("ffbuffersink");
    }
    CHECK(buffersrc) << "Error: no buffersrc";
    CHECK(buffersink) << "Error: no buffersink";
    AVFilterInOut *outputs = avfilter_inout_alloc();
	AVFilterInOut *inputs  = avfilter_inout_alloc();
	// AVBufferSinkParams *buffersink_params;

	filter_graph_.reset(avfilter_graph_alloc());
	/* set threads to 1, details see https://github.com/dmlc/decord/pull/63 */
	//LOG(INFO) << "Original GraphFilter nb_threads: " << filter_graph_->nb_threads;
	filter_graph_->nb_threads = 1;
    /* buffer video source: the decoded frames from the decoder will be inserted here. */
    // Sanitize sample_aspect_ratio: a zero denominator causes inf which FFmpeg 7+ rejects
    int sar_num = dec_ctx->sample_aspect_ratio.num;
    int sar_den = dec_ctx->sample_aspect_ratio.den;
    if (sar_den == 0) {
        sar_num = 1;
        sar_den = 1;
    }
#if LIBAVFILTER_VERSION_MAJOR >= 10
    // FFmpeg 7+: pix_fmt option uses AV_OPT_TYPE_PIXEL_FMT, requiring a format name string
    const char *pix_fmt_name = av_get_pix_fmt_name(dec_ctx->pix_fmt);
    if (!pix_fmt_name) pix_fmt_name = "yuv420p";
    std::snprintf(args, sizeof(args),
            "video_size=%dx%d:pix_fmt=%s:time_base=%d/%d:pixel_aspect=%d/%d",
            dec_ctx->width, dec_ctx->height, pix_fmt_name,
            dec_ctx->time_base.num, dec_ctx->time_base.den,
            sar_num, sar_den);
#else
    std::snprintf(args, sizeof(args),
            "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
            dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
            dec_ctx->time_base.num, dec_ctx->time_base.den,
            sar_num, sar_den);
#endif

    CHECK_GE(avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in",
		args, NULL, filter_graph_.get()), 0) << "Cannot create buffer source";

    /* buffer video sink: to terminate the filter chain. */
	CHECK_GE(avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out",
		NULL, NULL, filter_graph_.get()), 0) << "Cannot create buffer sink";
#if LIBAVFILTER_VERSION_MAJOR < 10
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGB24 , AV_PIX_FMT_NONE };
    CHECK_GE(av_opt_set_int_list(buffersink_ctx_, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN), 0) << "Set output pixel format error.";
#else
    // FFmpeg 7+: pix_fmts is no longer a runtime option on buffersink,
    // so enforce output format via the filter chain instead.
    filters_descr += ",format=rgb24";
#endif

    // LOG(INFO) << "create filter set opt";
    /* Endpoints for the filter graph. */
	outputs->name       = av_strdup("in");
	outputs->filter_ctx = buffersrc_ctx_;
	outputs->pad_idx    = 0;
	outputs->next       = NULL;

	inputs->name       = av_strdup("out");
	inputs->filter_ctx = buffersink_ctx_;
	inputs->pad_idx    = 0;
	inputs->next       = NULL;

    /* Parse filter description */
    CHECK_GE(avfilter_graph_parse_ptr(filter_graph_.get(), filters_descr.c_str(),
		&inputs, &outputs, NULL), 0) << "Failed to parse filters description.";

    /* Config filter graph */
    CHECK_GE(avfilter_graph_config(filter_graph_.get(), NULL), 0) << "Failed to config filter graph";

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
}

void FFMPEGFilterGraph::Push(AVFrame *frame) {
    // push decoded frame into filter graph
    CHECK_GE(av_buffersrc_add_frame_flags(buffersrc_ctx_, frame, AV_BUFFERSRC_FLAG_KEEP_REF), 0)
        << "Error while feeding the filter graph";
    ++count_;
}

bool FFMPEGFilterGraph::Pop(AVFrame **frame) {
    if (!count_.load()) {
        // LOG(INFO) << "No count in filter graph.";
        return false;
    }
    if (!*frame) *frame = av_frame_alloc();
    int ret = av_buffersink_get_frame(buffersink_ctx_, *frame);
    if (ret < 0) LOG(INFO) << "buffersink get frame failed" << AVERROR(ret);
    return ret >= 0;
}

}  // namespace ffmpeg
}  // namespace decord
