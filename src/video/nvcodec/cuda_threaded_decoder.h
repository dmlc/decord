/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_

#include "cuda_parser.h"
#include "cuda_decoder_impl.h"
#include "cuda_storage_pool.h"
#include "../ffmpeg/ffmpeg_common.h"

#include <condition_variable>
#include <thread>

#include <decord/runtime/ndarray.h>

namespace decord {
namespace cuda {

struct NumberedFrame {
    DLTensor t;
    int64_t n;
}

class CUThreadedDecoder {
    constexpr int kMaxOutputSurfaces = 20;
    using NDArray = runtime::NDArray;
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPARSERDISPINFO*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NumberedFrame>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;

    public:
        CUThreadedDecoder(int device_id);
        void SetCodecContext(const AVCodecContext *dec_ctx, int width = -1, int height = -1);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt, DLTensor buf);
        bool Pop(DLTensor *frame);
        ~CUThreadedDecoder();

        static int CUDAAPI HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI HandlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI HandlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        int HandlePictureSequence_(CUVIDEOFORMAT* format);
        int HandlePictureDecode_(CUVIDPICPARAMS* pic_params);
        int HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info);
        void LaunchThread();
        void ConvertThread();

        int device_id_;
        CUStream stream_;
        CUdevice device_;
        CUContext ctx_;
        CUVideoParser parser_;
        CUVideoDecoderImpl decoder_;
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        BufferQueuePtr buffer_queue_;
        std::unordered_map<int64_t, runtime::NDArray> reorder_buffer_;
        std::queue<int> surface_order_;  // read by main thread only, write by fetcher only
        std::thread launcher_t_;
        std::thread converter_t_;
        std::vector<dmlc::ConcurrentBlockingQueue<uint8_t>> permits_;
        std::atomic<bool> run_;
        std::atomic<int> frame_count_;
        CUTextureRegistry tex_registry_;
        AVRational nv_time_base_ = {1, 10000000};
        AVRational frame_base_;
        AVCodecContextPtr dec_ctx_;
    
    DISALLOW_COPY_AND_ASSIGN(CUThreadedDecoder);
};

int CUDAAPI CUThreadedDecoder::HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureSequence_(format);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDecode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureDecode_(pic_params);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDisplay(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->HandlePictureDisplay_(disp_info);
}

}  // namespace decord
}  // namespace cuda
#endif  // DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_