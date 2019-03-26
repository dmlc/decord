/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.cc
 * \brief FFmpeg video reader Impl
 */

#include "video_reader.h"

#include <decord/runtime/ndarray.h>

namespace decord {
namespace ffmpeg {

using NDArray = runtime::NDArray;

NDArray CopyToNDArray(AVFrame *p) {
    CHECK(p) << "Error: converting empty AVFrame to DLTensor";
    // int channel = p->linesize[0] / p->width;
    CHECK_EQ(AVPixelFormat(p->format), AV_PIX_FMT_RGB24) 
        << "Only support RGB24 image to NDArray conversion, given: " 
        << AVPixelFormat(p->format);
    
    DLContext ctx;
    if (p->hw_frames_ctx) {
        ctx = DLContext({kDLGPU, 0});
    } else {
        ctx = kCPU;
    }
    // LOG(INFO) << p->height << " x";
    DLTensor dlt;
    std::vector<int64_t> shape = {p->height, p->width, p->linesize[0] / p->width};
    dlt.data = p->data[0];
    dlt.ctx = ctx;
    dlt.ndim = 3;
    dlt.dtype = kUInt8;
    dlt.shape = dmlc::BeginPtr(shape);
    dlt.strides = NULL;
    dlt.byte_offset = 0;
    NDArray arr = NDArray::Empty({p->height, p->width, p->linesize[0] / p->width}, kUInt8, ctx);
    arr.CopyFrom(&dlt);
    return arr;
}

FFMPEGVideoReader::FFMPEGVideoReader(std::string fn, int width, int height)
     : codecs_(), actv_stm_idx_(-1), decoder_(), width_(width), height_(height), eof_(false) {
    // allocate format context
    fmt_ctx_.reset(avformat_alloc_context());
    if (!fmt_ctx_) {
        LOG(FATAL) << "ERROR allocating memory for Format Context";
    }
    // LOG(INFO) << "opened fmt ctx";
    // open file
    auto fmt_ctx = fmt_ctx_.get();
    if(avformat_open_input(&fmt_ctx, fn.c_str(), NULL, NULL) != 0 ) {
        LOG(FATAL) << "ERROR opening file: " << fn;
    }

    LOG(INFO) << "opened input";

    // find stream info
    if (avformat_find_stream_info(fmt_ctx,  NULL) < 0) {
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
    // avformat_free_context(fmt_ctx_);
    // avformat_close_input(&fmt_ctx_);
    LOG(INFO) << "Destruct Video REader";
}

void FFMPEGVideoReader::SetVideoStream(int stream_nb) {
    CHECK(fmt_ctx_ != NULL);
    AVCodec *dec;
    int st_nb = av_find_best_stream(fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, stream_nb, -1, &dec, 0);
    LOG(INFO) << "find best stream: " << st_nb;
    CHECK_GE(st_nb, 0) << "ERROR cannot find video stream with wanted index: " << stream_nb;
    // initialize the mem for codec context
    CHECK(codecs_[st_nb] == dec) << "Codecs of " << st_nb << " is NULL";
    LOG(INFO) << "codecs of stream: " << codecs_[st_nb] << " name: " <<  codecs_[st_nb]->name;
    decoder_ = std::unique_ptr<FFMPEGThreadedDecoder>(new FFMPEGThreadedDecoder());
    auto dec_ctx = avcodec_alloc_context3(dec);
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
    char descr[128];
    std::snprintf(descr, sizeof(descr),
            "scale=%d:%d", width_, height_);
    decoder_->SetCodecContext(dec_ctx, std::string(descr));
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
                << st->codecpar->width << "x" << st->codecpar->height
                << " Frame count: "
                << st->nb_frames;
        } else {
            const char *codec_type = av_get_media_type_string(st->codecpar->codec_type);
            codec_type = codec_type? codec_type : "unknown type";
            LOG(INFO) << codec_type << " stream [" << i << "].";
        }
        
    }
    return fmt_ctx_->nb_streams;
}

void FFMPEGVideoReader::PushNext() {
    // AVPacket *packet = av_packet_alloc();
    AVPacketPtr packet = AVPacketPool::Get()->Acquire();
    int ret = -1;
    while (!eof_) {
        ret = av_read_frame(fmt_ctx_.get(), packet.get());
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                eof_ = true;
                return;
            } else {
                LOG(FATAL) << "Error: av_read_frame failed with " << AVERROR(ret);
            }
            return;
        }
        if (packet->stream_index == actv_stm_idx_) {
            // LOG(INFO) << "Packet index: " << packet->stream_index << " vs. " << actv_stm_idx_;
            // av_packet_unref(packet);
            break;
        }
    }
    // LOG(INFO) << "Successfully load packet";
    decoder_->Push(packet);
    // LOG(INFO) << "Pushed packet to decoder.";
}

NDArray FFMPEGVideoReader::NextFrame() {
    AVFramePtr frame;
    decoder_->Start();
    bool ret = false;
    while (!ret) {
        PushNext();
        ret = decoder_->Pop(&frame);
        if (!ret && eof_) {
            return NDArray::Empty({}, kUInt8, kCPU);
        }
    }
    
    // int ret = decoder_->Pop(&frame);
    // if (!ret) {
    //     return NDArray::Empty({}, kUInt8, kCPU);
    // }
    NDArray arr = CopyToNDArray(frame.get());
    // av_frame_free(&frame);
    return arr;
}

}  // namespace ffmpeg
}  // namespace decord