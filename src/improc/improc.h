/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc.h
 * \brief Image processing functions
 */

#ifndef DECORD_IMPROC_IMPROC_H_
#define DECORD_IMPROC_IMPROC_H_

#include <stdint.h>

namespace decord {
namespace cuda {

#ifdef DECORD_USE_CUDA

void ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
                  uint8_t* dst, cudaStream_t stream, uint16_t input_width, uint16_t input_height,
                  int output_width, int output_height);
#endif
}  // namespace imp
}  // namespace decord


#endif  // DECORD_IMPROC_IMPROC_H_
