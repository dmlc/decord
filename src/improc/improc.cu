/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc.cu
 * \brief CUDA image processing kernels
 */

#include "improc.h"
#include <cuda_fp16.h>
// #include <stdio.h>

namespace decord {
namespace cuda {
namespace detail {

template<typename T>
struct YUV {
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

template<typename YUV_T, typename RGB_T>
__device__ void yuv2rgb(const YUV<YUV_T>& yuv, RGB_T* rgb,
                        size_t stride, bool normalized) {
    auto mult = normalized ? 1.0f : 255.0f;
    auto y = (static_cast<float>(yuv.y) - 16.0f/255) * mult;
    auto u = (static_cast<float>(yuv.u) - 128.0f/255) * mult;
    auto v = (static_cast<float>(yuv.v) - 128.0f/255) * mult;

    auto& m = yuv2rgb_mat;

    // could get tricky with a lambda, but this branch seems faster
    float r, g, b;
    if (normalized) {
        r = clip(y*m[0] + u*m[1] + v*m[2], 1.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 1.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 1.0);
    } else {
        r = clip(y*m[0] + u*m[1] + v*m[2], 255.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 255.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 255.0);
    }

    rgb[0] = convert<RGB_T>(r);
    rgb[stride] = convert<RGB_T>(g);
    rgb[stride*2] = convert<RGB_T>(b);
}

template<typename T>
__global__ void process_frame_kernel(
    cudaTextureObject_t luma, cudaTextureObject_t chroma,
    T* dst, uint16_t input_width, uint16_t input_height,
    uint16_t output_width, uint16_t output_height, float fx, float fy) {

    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= output_width || dst_y >= output_height)
        return;

    auto src_x = static_cast<float>(dst_x) * fx;

    auto src_y = static_cast<float>(dst_y) * fy;

    YUV<float> yuv;
    yuv.y = tex2D<float>(luma, src_x + 0.5, src_y + 0.5);
    auto uv = tex2D<float2>(chroma, (src_x / 2) + 0.5, (src_y / 2) + 0.5);
    yuv.u = uv.x;
    yuv.v = uv.y;

    T* out = dst + (dst_x + dst_y * output_width) * 3;
    yuv2rgb(yuv, out, 1, false);
}
}  // namespace detail

int DivUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

void ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
    uint8_t* dst, cudaStream_t stream, uint16_t input_width, uint16_t input_height,
    int output_width, int output_height) {
    // resize factor
    auto fx = static_cast<float>(input_width) / output_width;
    auto fy = static_cast<float>(input_height) / output_height;

    dim3 block(32, 8);
    dim3 grid(DivUp(output_width, block.x), DivUp(output_height, block.y));

    detail::process_frame_kernel<<<grid, block, 0, stream>>>
            (luma, chroma, dst, input_width, input_height, output_width, output_height, fx, fy);
}
}  // namespace cuda
}  // namespace decord
