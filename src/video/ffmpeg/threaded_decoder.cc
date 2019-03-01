/*!
 *  Copyright (c) 2019 by Contributors
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), run_(false){
    LOG(INFO) << "ThreadedDecoder ctor: " << run_.load();
    
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
        pkt_queue_ = PacketQueuePtr(new PacketQueue());
        frame_queue_ = FrameQueuePtr(new FrameQueue());
        run_.store(true);
        auto t = std::thread(&FFMPEGThreadedDecoder::WorkerThread, this);
        std::swap(t_, t);
    }
}

void FFMPEGThreadedDecoder::Stop() {
    if (run_.load()) {
        pkt_queue_->SignalForKill();
        run_.store(false);
        frame_queue_->SignalForKill();
    }
    if (t_.joinable()) {
        LOG(INFO) << "joining";
        t_.join();
    }
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
}

void FFMPEGThreadedDecoder::Push(AVPacket *pkt) {
    pkt_queue_->Push(pkt);
    ++frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(AVFrame **frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.
    if (!frame_count_.load()) {
        LOG(INFO) << "No count!";
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
}

void FFMPEGThreadedDecoder::WorkerThread() {
    while (run_.load()) {
        // CHECK(filter_graph_) << "FilterGraph not initialized.";
        if (!filter_graph_) return;
        AVPacket *pkt;
        AVFrame *frame;
        int got_picture;
        bool ret = pkt_queue_->Pop(&pkt);
        if (!ret) {
            return;
        }
        // decode frame from packet
        // frame.Alloc();
        frame = av_frame_alloc();
        CHECK_GE(avcodec_send_packet(dec_ctx_, pkt), 0) << "Thread worker: Error sending packet.";
        got_picture = avcodec_receive_frame(dec_ctx_, frame);
        // avcodec_decode_video2(dec_ctx_.ptr.get(), frame.Get(), &got_picture, pkt.Get());
        if (got_picture >= 0) {
            // filter image frame (format conversion, scaling...)
            filter_graph_->Push(frame);
            CHECK(filter_graph_->Pop(&frame)) << "Error fetch filtered frame.";
            frame_queue_->Push(frame);
        } else {
            LOG(FATAL) << "Thread worker: Error decoding frame." << got_picture;
        }
        // free raw memories allocated with ffmpeg
        av_packet_unref(pkt);
    }
}

}  // namespace ffmpeg
}  // namespace decord