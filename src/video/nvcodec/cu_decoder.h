/*!
 *  Copyright (c) 2019 by Contributors
 * \file cu_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CU_DECODER_H_

#include "cu_parser.h"
#include "../ffmpeg/ffmpeg_common.h"

namespace decord {
namespace cuda {

class CUThreadedDecoder {
    public:
        CUThreadedDecoder();
        void SetCodecContext(AVCodecContext *dec_ctx, int height = -1, int width = -1);
        bool initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt);
        // bool Pop(AVFramePtr *frame);

        static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        CUVideoParser parser_;
}

}  // namespace decord
}  // namespace cuda
#endif