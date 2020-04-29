/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.h
 * \brief FFmpeg threaded decoder definition
 */

#ifndef DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_
#define DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_

#include "filter_graph.h"
#include "../threaded_decoder_interface.h"
#include <decord/runtime/ndarray.h>

#include <thread>
#include <unordered_set>
#include <mutex>

#include <dmlc/concurrency.h>

namespace decord {
namespace ffmpeg {

class FFMPEGThreadedDecoder final : public ThreadedDecoderInterface {
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FFMPEGFilterGraphPtr = std::shared_ptr<FFMPEGFilterGraph>;

    public:
        FFMPEGThreadedDecoder();
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1,
                             int rotation = 0);
        void Start();
        void Stop();
        void Clear();
        void Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf);
        bool Pop(runtime::NDArray *frame);
        void SuggestDiscardPTS(std::vector<int64_t> dts);
        void ClearDiscardPTS();
        ~FFMPEGThreadedDecoder();
    private:
        void WorkerThread();
        void WorkerThreadImpl();
        void RecordInternalError(std::string message);
        void CheckErrorStatus();
        void ProcessFrame(AVFramePtr p, NDArray out_buf);
        NDArray CopyToNDArray(AVFramePtr p);
        NDArray AsNDArray(AVFramePtr p);
        // void FetcherThread(std::condition_variable& cv, FrameQueuePtr frame_queue);
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        BufferQueuePtr buffer_queue_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;
        std::thread t_;
        // std::thread fetcher_;
        // std::condition_variable cv_;
        std::atomic<bool> run_;
        FFMPEGFilterGraphPtr filter_graph_;
        AVCodecContextPtr dec_ctx_;
        std::unordered_set<int64_t> discard_pts_;
        std::mutex pts_mutex_;
        std::mutex error_mutex_;
        std::atomic<bool> error_status_;
        std::string error_message_;

    DISALLOW_COPY_AND_ASSIGN(FFMPEGThreadedDecoder);
};

}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_
