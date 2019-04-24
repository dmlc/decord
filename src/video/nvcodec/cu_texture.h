/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_texture.h
 * \brief NVCUVID texture objects
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_TEXTURE_H_
#define DECORD_VIDEO_NVCODEC_CU_TEXTURE_H_

#include "nvcuvid/nvcuvid.h"

#include <dmlc/base.h>

namespace decord {
namespace cuda {

class CUTexture {
    public:
        CUTexture();
        CUTexture(const cudaResourceDesc* pResDesc,
                const cudaTextureDesc* pTexDesc,
                const cudaResourceViewDesc* pResViewDesc);
        ~CUTexture();
        CUTexture(CUTexture&& other);
        CUTexture& operator=(CUTexture&& other);
        CUTexture(const CUTexture&) = delete;
        CUTexture& operator=(const CUTexture&) = delete;
        operator cudaTextureObject_t() const;
    private:
        bool valid_;
        cudaTextureObject_t object_;
};  // class CUTexture

struct CUImageTexture {
    CUTexture luma;
    CUTexture chroma;
};  // struct CUImageTexture

}  // namespace cuda
}  // namespace decord

#endif