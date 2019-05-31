/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), draining_(false), run_(false) {
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height) {
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    // LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    // std::string descr = "scale=320:240";
    char descr[128];
    std::snprintf(descr, sizeof(descr),
            "scale=%d:%d", width, height);
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get()));
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::Start() {
    if (!run_.load()) {
        pkt_queue_.reset(new PacketQueue());
        frame_queue_.reset(new FrameQueue());
        buffer_queue_.reset(new BufferQueue());
        run_.store(true);
        auto t = std::thread(&FFMPEGThreadedDecoder::WorkerThread, this);
        std::swap(t_, t);
    }
}

void FFMPEGThreadedDecoder::Stop() {
    if (run_.load()) {
        if (pkt_queue_) {
            pkt_queue_->SignalForKill();
        }
        if (buffer_queue_) {
            buffer_queue_->SignalForKill();
        }
        run_.store(false);
        if (frame_queue_) {
            frame_queue_->SignalForKill();
        }
    }
    if (t_.joinable()) {
        // LOG(INFO) << "joining";
        t_.join();
    }
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
    if (dec_ctx_.get()) {
        avcodec_flush_buffers(dec_ctx_.get());
    }
    frame_count_.store(0);
    draining_.store(false);
}

// void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt) {
//     CHECK(run_.load());
//     if (!pkt) {
//         CHECK(!draining_.load()) << "Start draining twice...";
//         draining_.store(true);
//     }
//     pkt_queue_->Push(pkt);
//     ++frame_count_;
    
//     // LOG(INFO)<< "frame push: " << frame_count_;
//     // LOG(INFO) << "Pushed pkt to pkt_queue";
// }

void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt, runtime::NDArray buf) {
    CHECK(run_.load());
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }
    pkt_queue_->Push(pkt);
    buffer_queue_->Push(buf);
    ++frame_count_;
    
    // LOG(INFO)<< "frame push: " << frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

// void FFMPEGThreadedDecoder::Skip(AVPacketPtr pkt) {
//    CHECK(run_.load());
//    if (!pkt) {
//         if (!draining_.load()) {
//             draining_.store(true);
//         }
//    } else {
//         CHECK(pkt->side_data == nullptr);
//         AVDictionary * frameDict = nullptr;
//         av_dict_set(&frameDict, "discard", std::to_string(1).c_str(), 0);
//         int frameDictSize = 0;
//         uint8_t *frameDictData = av_packet_pack_dictionary(frameDict, &frameDictSize);
//         av_dict_free(&frameDict);
//         av_packet_add_side_data(pkt.get(), AVPacketSideDataType::AV_PKT_DATA_STRINGS_METADATA, frameDictData, frameDictSize);
//    }
//    pkt_queue_->Push(pkt);
// }

// bool FFMPEGThreadedDecoder::Pop(AVFramePtr *frame) {
//     // Pop is blocking operation
//     // unblock and return false if queue has been destroyed.
    
//     if (!frame_count_.load() && !draining_.load()) {
//         return false;
//     }
//     bool ret = frame_queue_->Pop(frame);
    
//     if (ret){
//         --frame_count_;
//     }
//     return (ret && (*frame));
// }

bool FFMPEGThreadedDecoder::Pop(runtime::NDArray *frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.
    
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    bool ret = frame_queue_->Pop(frame);
    
    if (ret) {
        --frame_count_;
    }
    return (ret && frame->data_);
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
        if (!pkt) {
            // LOG(INFO) << "Draining mode start...";
            // draining mode, pulling buffered frames out
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), NULL), 0) << "Thread worker: Error entering draining mode.";
            while (true) {
                got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
                if (got_picture == AVERROR_EOF) break;
                // filter image frame (format conversion, scaling...)
                filter_graph_->Push(frame.get());
                AVFramePtr out_frame = AVFramePool::Get()->Acquire();
                AVFrame *out_frame_p = out_frame.get();
                CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                auto tmp = AsNDArray(out_frame);
                if (out_buf.defined()) {
                    CHECK(out_buf.Size() == tmp.Size());
                    out_buf.CopyFrom(tmp);
                    frame_queue_->Push(out_buf);
                } else {
                    frame_queue_->Push(tmp);
                }
            }
            frame_queue_->Push(NDArray());
        } else {
            // normal mode, push in valid packets and retrieve frames
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), pkt.get()), 0) << "Thread worker: Error sending packet.";
            got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
            if (got_picture == 0) {
                frame->pts = frame->best_effort_timestamp;
                if (pkt->side_data) {
                    frame_queue_->Push(NDArray::Empty({1}, kUInt8, kCPU));
                }
                // filter image frame (format conversion, scaling...)
                filter_graph_->Push(frame.get());
                AVFramePtr out_frame = AVFramePool::Get()->Acquire();
                AVFrame *out_frame_p = out_frame.get();
                CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                auto tmp = AsNDArray(out_frame);
                if (out_buf.defined()) {
                    CHECK(out_buf.Size() == tmp.Size());
                    out_buf.CopyFrom(tmp);
                    frame_queue_->Push(out_buf);
                } else {
                    frame_queue_->Push(tmp);
                }
                // LOG(INFO) << "pts: " <<out_frame->pts;
            } else if (AVERROR(EAGAIN) == got_picture || AVERROR_EOF == got_picture) {
                frame_queue_->Push(NDArray());
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