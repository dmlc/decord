/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder_impl.h
 * \brief NVCUVID based decoder implementation class
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_DECODER_IMPL_H_
#define DECORD_VIDEO_NVCODEC_CUDA_DECODER_IMPL_H_

#include "nvcuvid/nvcuvid.h"
#include <stdint.h>

namespace decord {
namespace cuda {

class CUVideoDecoderImpl {
  public:
    CUVideoDecoderImpl();
    CUVideoDecoderImpl(CUvideodecoder d);
    ~CUVideoDecoderImpl();

    // no copying
    CUVideoDecoderImpl(const CUVideoDecoderImpl&) = delete;
    CUVideoDecoderImpl& operator=(const CUVideoDecoderImpl&) = delete;

    CUVideoDecoderImpl(CUVideoDecoderImpl&& other);
    CUVideoDecoderImpl& operator=(CUVideoDecoderImpl&& other);

    operator CUvideodecoder() const;

    int Initialize(CUVIDEOFORMAT* format);
    bool Initialized() const;

    uint16_t Width() const;
    uint16_t Height() const;

  private:
    CUvideodecoder decoder_;
    CUVIDDECODECREATEINFO decoder_info_;
    bool initialized_;
};  // class CUVideoDecoderImpl

}  // namespace cuda
}  // namespace decord
#endif
