/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_parser.h
 * \brief NVCUVID based video parser
 */
#include "cuda_parser.h"
#include "cuda_threaded_decoder.h"


namespace decord {
namespace cuda {

void CUVideoParser::InitParams(AVCodecID codec, CUThreadedDecoder* decoder, int decode_surfaces,
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
        case AV_CODEC_ID_MPEG4:
            parser_info_.CodecType = cudaVideoCodec_MPEG4;
            parser_info_.ulMaxNumDecodeSurfaces = 20;
            break;
        case AV_CODEC_ID_VP9:
            parser_info_.CodecType = cudaVideoCodec_VP9;
            parser_info_.ulMaxNumDecodeSurfaces = 20;
            break;
        default:
            LOG(FATAL) << "Invalid codec: " << avcodec_get_name(AVCodecID(codec));
            return;
    }
    parser_info_.ulMaxNumDecodeSurfaces = decode_surfaces;
    parser_info_.ulErrorThreshold = 0;
    parser_info_.ulMaxDisplayDelay = 0;
    parser_info_.pUserData = decoder;
    parser_info_.pfnSequenceCallback = CUThreadedDecoder::HandlePictureSequence;
    parser_info_.pfnDecodePicture = CUThreadedDecoder::HandlePictureDecode;
    parser_info_.pfnDisplayPicture = CUThreadedDecoder::HandlePictureDisplay;
    parser_info_.pExtVideoInfo = &parser_extinfo_;
    if (extradata_size > 0) {
        auto hdr_size = std::min(sizeof(parser_extinfo_.raw_seqhdr_data),
                                    static_cast<std::size_t>(extradata_size));
        parser_extinfo_.format.seqhdr_data_length = hdr_size;
        memcpy(parser_extinfo_.raw_seqhdr_data, extradata, hdr_size);
    }
}

}  // namespace cuda
}  // namespace decord
