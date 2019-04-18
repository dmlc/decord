/*!
 *  Copyright (c) 2019 by Contributors
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), draining_(false), run_(false){
    // LOG(INFO) << "ThreadedDecoder ctor: " << run_.load();
    
    // Start();
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, std::string descr) {
    // LOG(INFO) << "Enter setcontext";
    bool running = run_.load();
    Stop();
    dec_ctx_.reset(dec_ctx);
    // LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    // std::string descr = "scale=320:240";
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get()));
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::Start() {
    if (!run_.load()) {
        pkt_queue_.reset(new PacketQueue());
        frame_queue_.reset(new FrameQueue());
        avcodec_flush_buffers(dec_ctx_.get());
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
        // LOG(INFO) << "joining";
        t_.join();
    }
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
}

void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt) {
    pkt_queue_->Push(pkt);
    ++frame_count_;
    if (!pkt) {
        draining_.store(true);
    }
    // LOG(INFO)<< "frame push: " << frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(AVFramePtr *frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.
    
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    bool ret = frame_queue_->Pop(frame);
    
    if (ret){
        --frame_count_;
    }
    return (ret && (*frame));
}

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
}

void FFMPEGThreadedDecoder::WorkerThread() {
    while (run_.load()) {
        // CHECK(filter_graph_) << "FilterGraph not initialized.";
        if (!filter_graph_) return;
        AVPacketPtr pkt;
        
        int got_picture;
        bool ret = pkt_queue_->Pop(&pkt);
        if (!ret) {
            return;
        }
        AVFramePtr frame = AVFramePool::Get()->Acquire();
        AVFramePtr out_frame = AVFramePool::Get()->Acquire();
        AVFrame *out_frame_p = out_frame.get();
        if (!pkt) {
            // LOG(INFO) << "Draining mode start...";
            // draining mode, pulling buffered frames out
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), NULL), 0) << "Thread worker: Error entering draining mode.";
            while (true) {
                got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
                if (got_picture == AVERROR_EOF) break;
                // filter image frame (format conversion, scaling...)
                filter_graph_->Push(frame.get());
                CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";
                frame_queue_->Push(out_frame);
            }
            draining_.store(false);
            frame_queue_->Push(AVFramePtr(nullptr, [](AVFrame *p){}));
        } else {
            // normal mode, push in valid packets and retrieve frames
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), pkt.get()), 0) << "Thread worker: Error sending packet.";
            got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
            if (got_picture == 0) {
                // filter image frame (format conversion, scaling...)
                filter_graph_->Push(frame.get());
                CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";
                frame_queue_->Push(out_frame);
                // LOG(INFO) << "pts: " << out_frame->pts;
            } else if (AVERROR(EAGAIN) == got_picture || AVERROR_EOF == got_picture) {
                frame_queue_->Push(AVFramePtr(nullptr, [](AVFrame *p){}));
            } else {
                LOG(FATAL) << "Thread worker: Error decoding frame: " << got_picture;
            }
        }
        // free raw memories allocated with ffmpeg
        // av_packet_unref(pkt);
    }
}

}  // namespace ffmpeg
}  // namespace decord