/*!
 *  Copyright (c) 2019 by Contributors
 * \file cu_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CU_DECODER_H_

#include "cu_parser.h"

namespace decord {
namespace cuda {

class CUDecoder {
    public:
        CUDecoder();
        bool initialized() const;

        static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);
}

}  // namespace decord
}  // namespace cuda
#endif