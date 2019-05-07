/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_

#include "cuda_stream.h"
#include "cuda_parser.h"
#include "cuda_context.h"
#include "cuda_decoder_impl.h"
#include "cuda_texture.h"
#include "../ffmpeg/ffmpeg_common.h"
#include "../threaded_decoder_interface.h"

#include <condition_variable>
#include <thread>

#include <decord/runtime/ndarray.h>
#include <dmlc/concurrency.h>
#include <dlpack/dlpack.h>

namespace decord {
namespace cuda {

class CUThreadedDecoder : public ThreadedDecoderInterface {
    constexpr static int kMaxOutputSurfaces = 20;
    using NDArray = runtime::NDArray;
    using AVPacketPtr = ffmpeg::AVPacketPtr;
    using AVCodecContextPtr = ffmpeg::AVCodecContextPtr;
    // using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueue = std::queue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPARSERDISPINFO*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using PermitQueue = dmlc::ConcurrentBlockingQueue<int>;
    using PermitQueuePtr = std::shared_ptr<PermitQueue>;
    using ReorderQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using ReorderQueuePtr = std::unique_ptr<ReorderQueue>;

    public:
        CUThreadedDecoder(int device_id);
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt, NDArray buf);
        bool Pop(NDArray *frame);
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
        ReorderQueuePtr reorder_queue_;
        std::queue<long int> frame_order_;
        std::thread launcher_t_;
        std::thread converter_t_;
        std::vector<PermitQueuePtr> permits_;
        std::atomic<bool> run_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;
        CUTextureRegistry tex_registry_;
        AVRational nv_time_base_;
        AVRational frame_base_;
        AVCodecContextPtr dec_ctx_;
        unsigned int width_;
        unsigned int height_;
    
    DISALLOW_COPY_AND_ASSIGN(CUThreadedDecoder);
};
}  // namespace cuda
}  // namespace decord
#endif  // DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_