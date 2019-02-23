/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.cpp
 * \brief FFMPEG implementations
 */

#include "ffmpeg.h"
#include <cstdio>
#include <memory>
#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {
using NDArray = runtime::NDArray;

// DLTensor AVFrame_::ToDLTensor() {
//     DLTensor dlt;
//     AVFrame *p = Get();
//     CHECK(p) << "Error: converting empty AVFrame to DLTensor";
//     shape[0] = p->height;
//     shape[1] = p->width;
//     shape[2] = p->linesize[0] / p->width;
//     CHECK_EQ(p->linesize[0], p->width * shape[2]) << "No Planar AVFrame to DLTensor support.";
//     dlt.data = p->data[0];
//     if (p->hw_frames_ctx) {
//         dlt.ctx = DLContext({kDLGPU, 0});
//     } else {
//         dlt.ctx = kCPU;
//     }
//     dlt.ndim = 3;
//     dlt.dtype = kUInt8;
//     dlt.shape = shape;
//     dlt.strides = NULL;
//     dlt.byte_offset = 0;
//     return dlt;
// }

NDArray CopyToNDArray(AVFrame *p) {
    CHECK(p) << "Error: converting empty AVFrame to DLTensor";
    // int channel = p->linesize[0] / p->width;
    CHECK_EQ(AVPixelFormat(p->format), AV_PIX_FMT_RGB24) << "Only support RGB24 image to NDArray conversion";
    DLContext ctx;
    if (p->hw_frames_ctx) {
        ctx = DLContext({kDLGPU, 0});
    } else {
        ctx = kCPU;
    }
    DLManagedTensor dlt;
    std::vector<int64_t> shape = {p->height, p->width, p->linesize[0] / p->width};
    dlt.dl_tensor.data = p->data[0];
    dlt.dl_tensor.ctx = ctx;
    dlt.dl_tensor.ndim = 3;
    dlt.dl_tensor.dtype = kUInt8;
    dlt.dl_tensor.shape = dmlc::BeginPtr(shape);
    dlt.dl_tensor.strides = NULL;
    dlt.dl_tensor.byte_offset = 0;
    return NDArray::FromDLPack(&dlt);
}

FrameTransform::FrameTransform(DLDataType dtype, uint32_t h, uint32_t w, uint32_t c, int interp) 
        : height(h), width(w), channel(c), interp(interp) {
    CHECK(c == 3 || c == 1)
        << "Only 3 channel RGB or 1 channel Gray image format supported";
    if (dtype == kUInt8) {
        fmt = c == 3 ? AV_PIX_FMT_RGB24: AV_PIX_FMT_GRAY8;
    } else if (dtype == kUInt16) {
        fmt = c == 3 ? AV_PIX_FMT_RGB48: AV_PIX_FMT_GRAY16;
    } else if (dtype == kFloat16) {
        // FFMPEG has no native support of float formats
        // use fp16 and cast later
        fmt = c == 3 ? AV_PIX_FMT_RGB48: AV_PIX_FMT_GRAY16;
    } else {
        LOG(FATAL) << "Unsupported data type [" 
            << dtype.code << " " << dtype.bits << " " << dtype.lanes
            << " and channel combination: " << channel;
    }
};

// Constructor
FFMPEGVideoReader::FFMPEGVideoReader(std::string fn)
     : codecs_(), actv_stm_idx_(-1), fmt_ctx_(NULL), decoder_()  {
    // allocate format context
    fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
        LOG(FATAL) << "ERROR allocating memory for Format Context";
    }
    LOG(INFO) << "opened fmt ctx";
    // open file
    if(avformat_open_input(&fmt_ctx_, fn.c_str(), NULL, NULL) != 0 ) {
        LOG(FATAL) << "ERROR opening file: " << fn;
    }

    LOG(INFO) << "opened input";

    // find stream info
    if (avformat_find_stream_info(fmt_ctx_,  NULL) < 0) {
        LOG(FATAL) << "ERROR getting stream info of file" << fn;
    }

    LOG(INFO) << "find stream info";

    // initialize all video streams and store codecs info
    for (uint32_t i = 0; i < fmt_ctx_->nb_streams; ++i) {
        AVStream *st = fmt_ctx_->streams[i];
        AVCodec *local_codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // store video stream codecs only
            codecs_.emplace_back(local_codec);
        } else {
            // audio, subtitle... skip
            AVCodec *tmp = NULL;
            codecs_.emplace_back(tmp);
        }
    }
    LOG(INFO) << "initialized all streams";
    // find best video stream (-1 means auto, relay on FFMPEG)
    SetVideoStream(-1);
    LOG(INFO) << "Set video stream";
    decoder_->Start();

    // // allocate AVFrame buffer
    // frame_ = av_frame_alloc();
    // CHECK(frame_) << "ERROR failed to allocated memory for AVFrame";

    // // allocate AVPacket buffer
    // pkt_ = av_packet_alloc();
    // CHECK(pkt_) << "ERROR failed to allocated memory for AVPacket";
}

FFMPEGVideoReader::~FFMPEGVideoReader(){
    avformat_free_context(fmt_ctx_);
}

void FFMPEGVideoReader::SetVideoStream(int stream_nb) {
    CHECK(fmt_ctx_ != NULL);
    int st_nb = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    LOG(INFO) << "find best stream: " << st_nb;
    CHECK_GE(st_nb, 0) << "ERROR cannot find video stream with wanted index: " << stream_nb;
    // initialize the mem for codec context
    CHECK(codecs_[st_nb] != NULL) << "Codecs of " << st_nb << " is NULL";
    LOG(INFO) << "codecs of stream: " << codecs_[st_nb] << " name: " <<  codecs_[st_nb]->name;
    decoder_ = std::unique_ptr<FFMPEGThreadedDecoder>(new FFMPEGThreadedDecoder());
    auto dec_ctx = avcodec_alloc_context3(codecs_[st_nb]);
    // CHECK_GE(avcodec_copy_context(dec_ctx, fmt_ctx_->streams[stream_nb]->codec), 0) << "Error: copy context";
    // CHECK_GE(avcodec_parameters_to_context(dec_ctx, fmt_ctx_->streams[st_nb]->codecpar), 0) << "Error: copy parameters to codec context.";
    // copy codec parameters to context
    CHECK_GE(avcodec_parameters_to_context(dec_ctx, fmt_ctx_->streams[st_nb]->codecpar), 0)
        << "ERROR copying codec parameters to context";
    // initialize AVCodecContext to use given AVCodec
    CHECK_GE(avcodec_open2(dec_ctx, codecs_[st_nb], NULL), 0)
        << "ERROR open codec through avcodec_open2";
    LOG(INFO) << "codecs opened.";
    actv_stm_idx_ = st_nb;
    LOG(INFO) << "time base: " << fmt_ctx_->streams[st_nb]->time_base.num << " / " << fmt_ctx_->streams[st_nb]->time_base.den;
    dec_ctx->time_base = fmt_ctx_->streams[st_nb]->time_base;
    decoder_->SetCodecContext(dec_ctx);
}

unsigned int FFMPEGVideoReader::QueryStreams() const {
    CHECK(fmt_ctx_ != NULL);
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; ++i) {
        // iterate and print stream info
        // feel free to add more if needed
        AVStream *st = fmt_ctx_->streams[i];
        AVCodec *codec = codecs_[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            LOG(INFO) << "video stream [" << i << "]:"
                << " Average FPS: " 
                << static_cast<float>(st->avg_frame_rate.num) / st->avg_frame_rate.den
                << " Start time: "
                << st->start_time
                << " Duration: "
                << st->duration
                << " Codec Type: "
                << codec->name
                << " ID: "
                << codec->id
                << " bit_rate: "
                << st->codecpar->bit_rate
                << " Resolution: "
                << st->codecpar->width << "x" << st->codecpar->height;
        } else {
            const char *codec_type = av_get_media_type_string(st->codecpar->codec_type);
            codec_type = codec_type? codec_type : "unknown type";
            LOG(INFO) << codec_type << " stream [" << i << "].";
        }
        
    }
    return fmt_ctx_->nb_streams;
}

void FFMPEGVideoReader::PushNext() {
    AVPacket *packet = av_packet_alloc();
    int ret = -1;
    while (ret <= 0) {
        ret = av_read_frame(fmt_ctx_, packet);
        if (ret < 0) {
            if (ret == AVERROR(EAGAIN)) {
                LOG(INFO) << "EAGAIN";
            } else if (ret == AVERROR_EOF) {
                LOG(FATAL) << "End of read file";
            } else {
                LOG(FATAL) << "Other error";
            }
            
            return;
        }
        if (packet->stream_index != actv_stm_idx_) {
            // LOG(INFO) << "Packet index: " << packet->stream_index << " vs. " << actv_stm_idx_;
            ret = -1;
            av_packet_unref(packet);
        }
    }
    LOG(INFO) << "Successfully load packet";
    decoder_->Push(packet);
    LOG(INFO) << "Pushed packet to decoder.";
}

NDArray FFMPEGVideoReader::NextFrame() {
    AVFrame *frame;
    decoder_->Start();
    PushNext();
    CHECK(decoder_->Pop(&frame)) << "Error getting next frame.";
    // DLTensor dlt = copytondarray(frame);
    // runtime::ndarray arr;
    // arr.copyfrom(&dlt);
    return CopyToNDArray(frame);
}

// struct SwsContext* FFMPEGVideoReader::GetSwsContext(FrameTransform out_fmt) {
//     auto sws_ctx = sws_ctx_map_.find(out_fmt);
//     if ( sws_ctx == sws_ctx_map_.end()) {
//         // already initialized sws context
//         return sws_ctx->second;
//     } else {
//         // create new sws context
//         struct SwsContext *ctx = sws_getContext(dec_ctx_->width,
//                                                 dec_ctx_->height,
//                                                 dec_ctx_->pix_fmt,
//                                                 out_fmt.width,
//                                                 out_fmt.height,
//                                                 out_fmt.fmt,
//                                                 out_fmt.interp,
//                                                 NULL,
//                                                 NULL,
//                                                 NULL
//                                                 );
//         sws_ctx_map_[out_fmt] = ctx;
//         return ctx;
//     }
// }

// void FFMPEGVideoDecoder::Reset() {
//     avformat_close_input(&fmt_ctx_);
//     avformat_free_context(fmt_ctx_);
//     av_packet_free(&pkt_);
//     av_frame_free(&frame_);
//     avcodec_free_context(&dec_ctx_);
//     for (auto& sws_ctx : sws_ctx_map_) {
//         sws_freeContext(sws_ctx.second);
//     }
//     sws_ctx_map_.clear();
// }

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), run_(false){
    LOG(INFO) << "ThreadedDecoder ctor: " << run_.load();
    pkt_queue_ = PacketQueuePtr(new PacketQueue());
    frame_queue_ = FrameQueuePtr(new FrameQueue());
    // Start();
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx) {
    LOG(INFO) << "Enter setcontext";
    bool running = run_.load();
    Stop();
    dec_ctx_ = dec_ctx;
    LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    std::string descr = "scale=320:240";
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx));
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::Start() {
    if (!run_.load()) {
        run_.store(true);
        auto t = std::thread(&FFMPEGThreadedDecoder::WorkerThread, this);
        std::swap(t_, t);
                        //  pkt_queue_, frame_queue_, filter_graph_, std::ref(run_));
    }
}

void FFMPEGThreadedDecoder::Stop() {
    if (run_.load()) {
        run_.store(false);
    }
    if (t_.joinable()) {
        t_.join();
    }
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
    pkt_queue_->SignalForKill();
    frame_queue_->SignalForKill();
    pkt_queue_ = PacketQueuePtr(new PacketQueue());
    frame_queue_ = FrameQueuePtr(new FrameQueue());
}

void FFMPEGThreadedDecoder::Push(AVPacket *pkt) {
    pkt_queue_->Push(pkt);
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(AVFrame **frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.
    if (!frame_count_.load()) {
        return false;
    }
    bool ret = frame_queue_->Pop(frame);
    if (ret){
        --frame_count_;
    }
    return ret;
}

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
    pkt_queue_->SignalForKill();
    frame_queue_->SignalForKill();
}

void FFMPEGThreadedDecoder::WorkerThread() {
    while (run_.load()) {
        // CHECK(filter_graph_) << "FilterGraph not initialized.";
        if (!filter_graph_) return;
        AVPacket *pkt;
        AVFrame *frame;
        int got_picture;
        bool ret = pkt_queue_->Pop(&pkt);
        if (!ret) return;
        LOG(INFO) << "Thread worker: packet received.";
        // decode frame from packet
        // frame.Alloc();
        frame = av_frame_alloc();
        CHECK_GE(avcodec_send_packet(dec_ctx_, pkt), 0) << "Thread worker: Error sending packet.";
        got_picture = avcodec_receive_frame(dec_ctx_, frame);
        // avcodec_decode_video2(dec_ctx_.ptr.get(), frame.Get(), &got_picture, pkt.Get());
        if (got_picture) {
            LOG(INFO) << "Thread worker: Frame decoded successfully!";
            // filter image frame (format conversion, scaling...)
            filter_graph_->Push(frame);
            LOG(INFO) << "Thread worker: Pushed frame to filterGraph";
            CHECK(filter_graph_->Pop(&frame)) << "Error fetch filtered frame.";
            frame_queue_->Push(frame);
            LOG(INFO) << "Thread worker: Pushed filtered frame to frame queue";
            ++frame_count_;
        } else {
            LOG(FATAL) << "Thread worker: Error decoding frame.";
        }
    }
}

FFMPEGFilterGraph::FFMPEGFilterGraph(std::string filters_descr, AVCodecContext *dec_ctx) : count_(0) {
    Init(filters_descr, dec_ctx);
}

void FFMPEGFilterGraph::Init(std::string filters_descr, AVCodecContext *dec_ctx) {
    char args[512];
    const AVFilter *buffersrc  = avfilter_get_by_name("buffer");
	const AVFilter *buffersink = avfilter_get_by_name("buffersink");
    CHECK(buffersrc) << "Error no buffersrc";
    CHECK(buffersink) << "Error no buffersink";
    AVFilterInOut *outputs = avfilter_inout_alloc();
	AVFilterInOut *inputs  = avfilter_inout_alloc();
	enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE };
	AVBufferSinkParams *buffersink_params;
 
	filter_graph_ = avfilter_graph_alloc();
    /* buffer video source: the decoded frames from the decoder will be inserted here. */
	std::snprintf(args, sizeof(args),
            "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
            dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
            dec_ctx->time_base.num, dec_ctx->time_base.den,
            dec_ctx->sample_aspect_ratio.num, dec_ctx->sample_aspect_ratio.den);
    // std::snprintf(args, sizeof(args),
    //         "video_size=%dx%d:pix_fmt=%d",
    //         dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt);
    
    LOG(INFO) << "filter args: " << args;
    
    CHECK_GE(avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in",
		args, NULL, filter_graph_), 0) << "Cannot create buffer source";

    /* buffer video sink: to terminate the filter chain. */
	buffersink_params = av_buffersink_params_alloc();
	buffersink_params->pixel_fmts = pix_fmts;
	CHECK_GE(avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out",
		NULL, buffersink_params, filter_graph_), 0) << "Cannot create buffer sink";
	av_free(buffersink_params);

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
    CHECK_GE(avfilter_graph_parse_ptr(filter_graph_, filters_descr.c_str(),
		&inputs, &outputs, NULL), 0) << "Failed to parse filters description.";

    /* Config filter graph */
    CHECK_GE(avfilter_graph_config(filter_graph_, NULL), 0) << "Failed to config filter graph";
}

void FFMPEGFilterGraph::Push(AVFrame *frame) {
    // push decoded frame into filter graph
    CHECK_GE(av_buffersrc_add_frame_flags(buffersrc_ctx_, frame, 0), 0) 
        << "Error while feeding the filter graph";
    ++count_;
}

bool FFMPEGFilterGraph::Pop(AVFrame **frame) {
    if (!count_.load()) return false;
    if (!*frame) *frame = av_frame_alloc();
    int ret = av_buffersink_get_frame(buffersink_ctx_, *frame);
    return ret > 0;
}

FFMPEGFilterGraph::~FFMPEGFilterGraph() {
    // avfilter_free(buffersink_ctx_);
    // avfilter_free(buffersrc_ctx_);
    avfilter_graph_free(&filter_graph_);
}


}  // namespace ffmpeg
}  // namespace decord