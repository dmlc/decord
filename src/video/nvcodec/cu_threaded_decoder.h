/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_THREADED_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CU_DECODER_H_

#include "cu_parser.h"
#include "cu_decoder_impl.h"
#include "../ffmpeg/ffmpeg_common.h"

#include <condition_variable>
#include <thread>

#include <decord/runtime/ndarray.h>

namespace decord {
namespace cuda {

struct NumberedFrame {
    NDArray arr;
    int64_t n;
}

class CUThreadedDecoder {
    using NDArray = runtime::NDArray;
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPICPARAMS*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NumberedFrame>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;

    public:
        CUThreadedDecoder(int device_id, const AVCodecContext *dec_ctx);
        void SetCodecContext(AVCodecContext *dec_ctx, int height = -1, int width = -1);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt);
        bool Pop(NDArray *frame);
        ~CUThreadedDecoder();

        static int CUDAAPI CallbackSequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI CallbackDecode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI CallbackDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        int CallbackSequence_(CUVIDEOFORMAT* format);
        int CallbackDecode_(CUVIDPICPARAMS* pic_params);
        int CallbackDisplay_(CUVIDPARSERDISPINFO* disp_info);
        void FetcherThread();
        void WorkerThread(int worker_idx);

        int device_id_;
        CUdevice device_;
        CUContext ctx_;
        CUVideoParser parser_;
        CUVideoDecoderImpl decoder_;
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        std::vector<BufferQueuePtr> buffer_queues_;
        std::unordered_map<int64_t, runtime::NDarray> reorder_buffer_;
        std::queue<int> surface_order_;  // read by main thread only, write by fetcher only
        std::thread fetcher_;
        std::vector<std::thread> workers_;
        std::vector<dmlc::ConcurrentBlockingQueue<uint8_t>> worker_permits_;
        std::atomic<bool> run_;
    
    DISALLOW_COPY_AND_ASSIGN(CUThreadedDecoder);
}；

int CUDAAPI CUThreadedDecoder::CallbackSequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->CallbackSequence_(format);
}

int CUDAAPI CUThreadedDecoder::CallbackDecode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->CallbackDecode_(pic_params);
}

int CUDAAPI CUThreadedDecoder::CallbackDisplay(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->CallbackDisplay_(disp_info);
}

}  // namespace decord
}  // namespace cuda
#endif