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
#include <decord/runtime/ndarray.h>

namespace decord {
namespace cuda {

class CUThreadedDecoder {
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPARSERDISPINFO*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<runtime::NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;

    public:
        CUThreadedDecoder(int device_id, const AVCodecContext *dec_ctx);
        void SetCodecContext(AVCodecContext *dec_ctx, int height = -1, int width = -1);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt);
        // bool Pop(AVFramePtr *frame);

        static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        int device_id_;
        CUdevice device_;
        CUContext ctx_;
        CUVideoParser parser_;
        CUVideoDecoderImpl decoder_;
        std::vector<uint8_t> occupied_frames_;
        PacketQueuePtr pkt_queue_;
        BufferQueuePtr buffer_queue_;
        FrameQueuePtr frame_queue_;
        std::thread decode_t_;
        std::thread convert_t_;
        std::atomic<bool> run_;
        
}；

int CUDAAPI CUThreadedDecoder::handle_sequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->handle_sequence_(format);
}

int CUDAAPI CUThreadedDecoder::handle_decode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->handle_decode_(pic_params);
}

int CUDAAPI CUThreadedDecoder::handle_display(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    return decoder->handle_display_(disp_info);
}

}  // namespace decord
}  // namespace cuda
#endif