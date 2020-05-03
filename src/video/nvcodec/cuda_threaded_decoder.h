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
#include <mutex>

#include <decord/runtime/ndarray.h>
#include <dmlc/concurrency.h>
#include <dlpack/dlpack.h>

namespace decord {
namespace cuda {

class CUThreadedDecoder final : public ThreadedDecoderInterface {
    constexpr static int kMaxOutputSurfaces = 20;
    using NDArray = runtime::NDArray;
    using AVPacketPtr = ffmpeg::AVPacketPtr;
    using AVCodecContextPtr = ffmpeg::AVCodecContextPtr;
    using AVBSFContextPtr = ffmpeg::AVBSFContextPtr;
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPARSERDISPINFO*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using PermitQueue = dmlc::ConcurrentBlockingQueue<int>;
    using PermitQueuePtr = std::shared_ptr<PermitQueue>;
    using ReorderQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using ReorderQueuePtr = std::unique_ptr<ReorderQueue>;
    using FrameOrderQueue = dmlc::ConcurrentBlockingQueue<int64_t>;
    using FrameOrderQueuePtr = std::unique_ptr<FrameOrderQueue>;

    public:
        CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, AVInputFormat *iformat);
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1, int rotation = 0);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt, NDArray buf);
        bool Pop(NDArray *frame);
        void SuggestDiscardPTS(std::vector<int64_t> dts);
        void ClearDiscardPTS();
        ~CUThreadedDecoder();

        static int CUDAAPI HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI HandlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI HandlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        int HandlePictureSequence_(CUVIDEOFORMAT* format);
        int HandlePictureDecode_(CUVIDPICPARAMS* pic_params);
        int HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info);
        void LaunchThread();
        void LaunchThreadImpl();
        void RecordInternalError(std::string message);
        void CheckErrorStatus();
        void InitBitStreamFilter(AVCodecParameters *codecpar, AVInputFormat *iformat);

        int device_id_;
        CUStream stream_;
        CUdevice device_;
        CUContext ctx_;
        CUVideoParser parser_;
        CUVideoDecoderImpl decoder_;
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        //BufferQueuePtr buffer_queue_;
        //std::unordered_map<int64_t, runtime::NDArray> reorder_buffer_;
        ReorderQueuePtr reorder_queue_;
        //FrameOrderQueuePtr frame_order_;
        // int64_t last_pts_;
        std::thread launcher_t_;
        //std::thread converter_t_;
        //std::vector<PermitQueuePtr> permits_;
        // std::vector<uint8_t> frame_in_use_;
        std::atomic<bool> run_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;

        CUTextureRegistry tex_registry_;
        AVRational nv_time_base_;
        AVRational frame_base_;
        AVCodecContextPtr dec_ctx_;
        /*! \brief AV bitstream filter context */
        AVBSFContextPtr bsf_ctx_;
        unsigned int width_;
        unsigned int height_;
        // uint64_t decoded_cnt_;
        std::unordered_set<int64_t> discard_pts_;
        std::mutex pts_mutex_;
        std::mutex error_mutex_;
        std::atomic<bool> error_status_;
        std::string error_message_;

    DISALLOW_COPY_AND_ASSIGN(CUThreadedDecoder);
};
}  // namespace cuda
}  // namespace decord
#endif  // DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_
