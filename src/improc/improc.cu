/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc.cu
 * \brief CUDA image processing kernels
 */

#include "../video/nvcodec/cu_utils.h"

#include <cuda_fp16.h>

namespace decord {
namespace cuda {
namespace detail {

// using math from https://msdn.microsoft.com/en-us/library/windows/desktop/dd206750(v=vs.85).aspx

template<typename T>
struct yuv {
    T y, u, v;
};

__constant__ float yuv2rgb_mat[9] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

__device__ float clip(float x, float max) {
    return fmin(fmax(x, 0.0f), max);
}

template<typename T>
__device__ T convert(const float x) {
    return static_cast<T>(x);
}

template<>
__device__ half convert<half>(const float x) {
    return __float2half(x);
}

template<>
__device__ uint8_t convert<uint8_t>(const float x) {
    return static_cast<uint8_t>(roundf(x));
}

int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}
}  // namespace detail
}  // namespace cuda
}  // namespace decord