/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef DECORD_RUNTIME_CUDA_CUDA_COMMON_H_
#define DECORD_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <decord/runtime/packed_func.h>
#include <string>
#include "../workspace_pool.h"

namespace decord {
namespace runtime {

#ifdef __cuda_cuda_h__
inline bool check_cuda_call(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(e, &err);
        std::cerr << "CUDA error " << e << " at line " << iLine << " in file " << szFile
                  << ": " << err << std::endl;
        return false;
    }
    return true;
}
#endif

#ifdef __CUDA_RUNTIME_H__
inline bool check_cuda_call(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA runtime error " << e << " at line " << iLine
                  << " in file " << szFile
                  << ": " << cudaGetErrorString(e)
                  << std::endl;
        return false;
    }
    return true;
}
#endif

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg             \
          << " at line: " << __LINE__ << " in file: " << __FILE__;      \
    }                                                                   \
  }

#define CUDA_CALL(func)                                                  \
  {                                                                      \
    cudaError_t e = (func);                                              \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)             \
        << "CUDA: " << cudaGetErrorString(e) << " at line: " << __LINE__ \
        << " in file: " << __FILE__;                                     \
  }

#define CHECK_CUDA_CALL(x) check_cuda_call(x, __LINE__, __FILE__)

/*! \brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace decord
#endif  // DECORD_RUNTIME_CUDA_CUDA_COMMON_H_
