/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_parser.h
 * \brief NVCUVID based parser
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_PARSER_H_
#define DECORD_VIDEO_NVCODEC_CU_PARSER_H_

#include <cstring>

#include "nvcuvid/nvcuvid.h"
#include "../../runtime/cuda/cuda_common.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif

namespace decord {
namespace cuda {

class CUVideoParser {
  public:
    CUVideoParser() : parser_{0}, initialized_{false} {}

    CUVideoParser(AVCodecID codec, CUThreadedDecoder* decoder, int decode_surfaces)
        : CUVideoParser{codec, decoder, decode_surfaces, nullptr, 0} {}

    CUVideoParser(AVCodecID codec, CUThreadedDecoder* decoder, int decode_surfaces,
                  uint8_t* extradata, int extradata_size)
        : parser_{0}, parser_info_{}, parser_extinfo_{}, initialized_{false}
    {
        InitParams(codec, decoder, decode_surfaces, extradata, extradata_size);

        CUDA_DRIVER_CALL(cuvidCreateVideoParser(&parser_, &parser_info_));
        initialized_ = true;
    }

    CUVideoParser(CUvideoparser parser)
        : parser_{parser}, initialized_{true}
    {
    }

    ~CUVideoParser() {
        if (initialized_) {
            CUDA_DRIVER_CALL(cuvidDestroyVideoParser(parser_));
        }
    }

    CUVideoParser(CUVideoParser&& other)
        : parser_{other.parser_}, initialized_{other.initialized_}
    {
        other.parser_ = 0;
        other.initialized_ = false;
    }

    CUVideoParser& operator=(CUVideoParser&& other) {
        if (initialized_) {
            CUDA_DRIVER_CALL(cuvidDestroyVideoParser(parser_));
        }
        parser_ = other.parser_;
        parser_info_ = other.parser_info_;
        parser_extinfo_ = other.parser_extinfo_;
        initialized_ = other.initialized_;
        other.parser_ = 0;
        other.initialized_ = false;
        return *this;
    }

    bool Initialized() const {
        return initialized_;
    }

    operator CUvideoparser() const {
        return parser_;
    }

  private:
    void InitParams(AVCodecID codec, CUThreadedDecoder* decoder, int decode_surfaces,
                    uint8_t* extradata, int extradata_size);
    CUvideoparser parser_;
    CUVIDPARSERPARAMS parser_info_;
    CUVIDEOFORMATEX parser_extinfo_;

    bool initialized_;
};

}  // namespace decord
}  // namespace cuda

#endif
