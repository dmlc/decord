/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_stream.h
 * \brief CUDA stream
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_STREAM_H_
#define DECORD_VIDEO_NVCODEC_CU_STREAM_H_

#include "cu_utils.h"

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