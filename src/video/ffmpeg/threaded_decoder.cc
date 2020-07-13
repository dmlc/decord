/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder() : frame_count_(0), draining_(false), run_(false), error_status_(false), error_message_() {
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation) {
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    // LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    // std::string descr = "scale=320:240";
    char descr[128];
    switch(rotation) {
        case 90:
            std::snprintf(descr, sizeof(descr), "transpose=1,scale=%d:%d", width, height);
            break;
        case 180:
            std::snprintf(descr, sizeof(descr), "transpose=1,transpose=1,scale=%d:%d", width, height);
            break;
        case 270:
            std::snprintf(descr, sizeof(descr), "transpose=2,scale=%d:%d", width, height);
            break;
        case 0:
        default:
            std::snprintf(descr, sizeof(descr), "scale=%d:%d", width, height);
    }
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get()));
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::Start() {
    CheckErrorStatus();
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
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      discard_pts_.clear();
    }
    error_status_.store(false);
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_.clear();
    }
}

void FFMPEGThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.insert(dts.begin(), dts.end());
}

void FFMPEGThreadedDecoder::ClearDiscardPTS() {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.clear();
}

void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt, runtime::NDArray buf) {
    CHECK(run_.load());
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }

    pkt_queue_->Push(pkt);
    buffer_queue_->Push(buf);

    // LOG(INFO)<< "frame push: " << frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(runtime::NDArray *frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.

    CheckErrorStatus();
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    // LOG(INFO) << "Waiting for pop";
    bool ret = frame_queue_->Pop(frame);
    // LOG(INFO) << "Poped";
    CheckErrorStatus();

    if (ret) {
        --frame_count_;
    }
    return (ret && frame->data_);
}

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
}

void FFMPEGThreadedDecoder::ProcessFrame(AVFramePtr frame, NDArray out_buf) {
    frame->pts = frame->best_effort_timestamp;
    bool skip = false;
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      skip = discard_pts_.find(frame->pts) != discard_pts_.end();
    }
    if (skip) {
        // skip resize/filtering
        NDArray empty = NDArray::Empty({1}, kUInt8, kCPU);
        empty.pts = frame->pts;
        frame_queue_->Push(empty);
        ++frame_count_;
        return;
    }
    // filter image frame (format conversion, scaling...)
    filter_graph_->Push(frame.get());
    AVFramePtr out_frame = AVFramePool::Get()->Acquire();
    AVFrame *out_frame_p = out_frame.get();
    CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";

    auto tmp = AsNDArray(out_frame);
    if (out_buf.defined()) {
        CHECK(out_buf.Size() == tmp.Size());
        out_buf.CopyFrom(tmp);
        frame_queue_->Push(out_buf);
        ++frame_count_;
    } else {
        frame_queue_->Push(tmp);
        ++frame_count_;
    }
}

void FFMPEGThreadedDecoder::WorkerThread() {
    try {
        WorkerThreadImpl();
    } catch (dmlc::Error error) {
        RecordInternalError(error.what());
        run_.store(false);
        frame_queue_->SignalForKill(); // Unblock all consumers
    }
}

void FFMPEGThreadedDecoder::WorkerThreadImpl() {
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
                if (got_picture == AVERROR_EOF) {
                    // LOG(INFO) << "stop draining";
                    for (int cnt = 0; cnt < 128; ++cnt) {
                        // special signal
                        frame_queue_->Push(NDArray::Empty({1}, kInt64, kCPU));
                        ++frame_count_;
                    }
                    draining_.store(false);
                    break;
                }
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                ProcessFrame(frame, out_buf);
            }
        } else {
            // normal mode, push in valid packets and retrieve frames
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), pkt.get()), 0) << "Thread worker: Error sending packet.";
            got_picture = avcodec_receive_frame(dec_ctx_.get(), frame.get());
            if (got_picture == 0) {
                NDArray out_buf;
                bool get_buf = buffer_queue_->Pop(&out_buf);
                if (!get_buf) return;
                ProcessFrame(frame, out_buf);
            } else if (AVERROR(EAGAIN) == got_picture || AVERROR_EOF == got_picture) {
                frame_queue_->Push(NDArray());
                ++frame_count_;
            } else {
                LOG(FATAL) << "Thread worker: Error decoding frame: " << got_picture;
            }
        }
        // free raw memories allocated with ffmpeg
        // av_packet_unref(pkt);
    }
}

NDArray FFMPEGThreadedDecoder::CopyToNDArray(AVFramePtr p) {
    CHECK(p) << "Error: converting empty AVFrame to DLTensor";
    // int channel = p->linesize[0] / p->width;
    CHECK(AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 || AVPixelFormat(p->format) == AV_PIX_FMT_GRAY8)
        << "Only support RGB24/GRAY8 image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    int channel = AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 ? 3 : 1;
    // CHECK(p->linesize[0] % p->width == 0)
    //     << "AVFrame data is not a compact array. linesize: " << p->linesize[0]
    //     << " width: " << p->width;

    DLContext ctx;
    CHECK(!p->hw_frames_ctx) << "Not supported hw_frames_ctx";
    ctx = kCPU;
    NDArray arr = NDArray::Empty({p->height, p->width, channel}, kUInt8, ctx);
    auto device_api = runtime::DeviceAPI::Get(ctx);
    void *to_ptr = arr.data_->dl_tensor.data;
    void *from_ptr = p->data[0];
    int linesize = p->width * channel;

    // arr.CopyFrom(&dlt);
    for (int i = 0; i < p->height; ++i) {
        // copy line by line
        device_api->CopyDataFromTo(
            from_ptr, i * p->linesize[0],
            to_ptr, i * linesize,
            linesize, ctx, ctx, kUInt8, nullptr);
    }
    arr.pts = p->pts;
    return arr;
}

static void AVFrameManagerDeleter(DLManagedTensor *manager) {
	delete static_cast<AVFrameManager*>(manager->manager_ctx);
	delete manager;
}

NDArray FFMPEGThreadedDecoder::AsNDArray(AVFramePtr p) {
    if (p->linesize[0] % p->width != 0) {
        // Fallback to copy since original AVFrame is not compact
        return CopyToNDArray(p);
    }
	DLManagedTensor* manager = new DLManagedTensor();
    auto av_manager = new AVFrameManager(p);
	manager->manager_ctx = av_manager;
	ToDLTensor(p, manager->dl_tensor, av_manager->shape);
	manager->deleter = AVFrameManagerDeleter;
	NDArray arr = NDArray::FromDLPack(manager);
    arr.pts = p->pts;
	return arr;
}

void FFMPEGThreadedDecoder::CheckErrorStatus() {
    if (error_status_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        LOG(FATAL) << error_message_;
    }
}

void FFMPEGThreadedDecoder::RecordInternalError(std::string message) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = message;
    }
    error_status_.store(true);
}

}  // namespace ffmpeg
}  // namespace decord
