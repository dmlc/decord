/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_common.h
 * \brief Cuda commons for video decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_STORAGE_POOL_H_
#define DECORD_VIDEO_NVCODEC_CUDA_STORAGE_POOL_H_

#include "../storage_pool.h"
#include "../../runtime/cuda/cuda_common.h"

namespace decord {
namespace cuda {

template<typename S, typename M>
class CUDAFixedSizeMemoryPool : public AutoReleasePool<void, S> {
    private:
        using T = void;
        T* Allocate() final {
            void* data = nullptr;
            CUDA_CALL(cudaMalloc(&data, M));
            return data;
        }

        void Delete(T* p) final {
            CUDA_CALL(cudaFree(p));
        }
};  // class CUDAFixedSizeMemoryPool

}  // namespace cuda
}  // namespace decord

#endif  // DECORD_VIDEO_NVCODEC_CUDA_STORAGE_POOL_H_