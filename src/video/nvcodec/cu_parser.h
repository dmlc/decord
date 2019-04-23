/*!
 *  Copyright (c) 2019 by Contributors
 * \file cu_parser.h
 * \brief NVCUVID based parser
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_PARSER_H_
#define DECORD_VIDEO_NVCODEC_CU_PARSER_H_

#include <cstring>

#include "nvcuvid/nvcuvid.h"
#include "cu_utils.h"

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

    template<typename Decoder>
    CUVideoParser(Codec codec, Decoder* decoder, int decode_surfaces)
        : CUVideoParser{codec, decoder, decode_surfaces, nullptr, 0} {}

    template<typename Decoder>
    CUVideoParser(Codec codec, Decoder* decoder, int decode_surfaces,
                  uint8_t* extradata, int extradata_size)
        : parser_{0}, parser_info_{}, parser_extinfo_{}, initialized_{false}
    {
        init_params(codec, decoder, decode_surfaces, extradata, extradata_size);

        if (CUDA_CHECK_CALL(cuvidCreateVideoParser(&parser_, &parser_info_))) {
            initialized_ = true;
        } else {
            LOG(FATAL) << "Problem creating video parser" << std::endl;
        }

    }

    template<typename Decoder>
    void init_params(int codec, Decoder* decoder, int decode_surfaces,
                     uint8_t* extradata, int extradata_size) {
        switch (codec) {
            case AV_CODEC_ID_H264:
                parser_info_.CodecType = cudaVideoCodec_H264;
                break;
            case AV_CODEC_ID_HEVC:
                parser_info_.CodecType = cudaVideoCodec_HEVC;
                // this can probably be better
                parser_info_.ulMaxNumDecodeSurfaces = 20;
                break;
            default:
                LOG(FATAL) << "Invalid codec: " << avcodec_get_name(codec);
                return;
        }
        parser_info_.ulMaxNumDecodeSurfaces = decode_surfaces;
        parser_info_.pUserData = decoder;
        parser_info_.pfnSequenceCallback = Decoder::handle_sequence;
        parser_info_.pfnDecodePicture = Decoder::handle_decode;
        parser_info_.pfnDisplayPicture = Decoder::handle_display;
        parser_info_.pExtVideoInfo = &parser_extinfo_;
        if (extradata_size > 0) {
            auto hdr_size = std::min(sizeof(parser_extinfo_.raw_seqhdr_data),
                                     static_cast<std::size_t>(extradata_size));
            parser_extinfo_.format.seqhdr_data_length = hdr_size;
            memcpy(parser_extinfo_.raw_seqhdr_data, extradata, hdr_size);
        }
    }

    CUVideoParser(CUvideoparser parser)
        : parser_{parser}, initialized_{true}
    {
    }

    ~CUVideoParser() {
        if (initialized_) {
            CUDA_CHECK_CALL(cuvidDestroyVideoParser(parser_));
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
            CUDA_CHECK_CALL(cuvidDestroyVideoParser(parser_));
        }
        parser_ = other.parser_;
        parser_info_ = other.parser_info_;
        parser_extinfo_ = other.parser_extinfo_;
        initialized_ = other.initialized_;
        other.parser_ = 0;
        other.initialized_ = false;
        return *this;
    }

    bool initialized() const {
        return initialized_;
    }

    operator CUvideoparser() const {
        return parser_;
    }

  private:
    CUvideoparser parser_;
    CUVIDPARSERPARAMS parser_info_;
    CUVIDEOFORMATEX parser_extinfo_;

    bool initialized_;
};

}  // namespace decord
}  // namespace cuda

#endif