/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_stream.h
 * \brief CUDA stream
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_STREAM_H_
#define DECORD_VIDEO_NVCODEC_CUDA_STREAM_H_

#include "../../runtime/cuda/cuda_common.h"

namespace decord {
namespace cuda {

class CUStream {
  public:
    CUStream(int device_id, bool default_stream);
    ~CUStream();
    CUStream(const CUStream&) = delete;
    CUStream& operator=(const CUStream&) = delete;
    CUStream(CUStream&&);
    CUStream& operator=(CUStream&&);
    operator cudaStream_t();

  private:
    bool created_;
    cudaStream_t stream_;
};  // class CUStream

}  // namespace cuda
}  // namespace decord
#endif
