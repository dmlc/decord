/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_texture.cc
 * \brief NVCUVID texture objects
 */

#include "cu_texture.h"

namespace decord {
namespace cuda {

CUTexture::CUTexture() : valid_{false} {
}

CUTexture::CUTexture(const cudaResourceDesc* pResDesc,
                     const cudaTextureDesc* pTexDesc,
                     const cudaResourceViewDesc* pResViewDesc)
    : valid_{false}
{
    if (!CUDA_CHECK_CALL(cudaCreateCUTexture(&object_, pResDesc, pTexDesc, pResViewDesc))) {
        LOG(FATAL) << "Unable to create a texture object";
    }
    valid_ = true;
}

CUTexture::~CUTexture() {
    if (valid_) {
        cudaDestroyCUTexture(object_);
    }
}

CUTexture::CUTexture(CUTexture&& other)
    : valid_{other.valid_}, object_{other.object_}
{
    other.valid_ = false;
}

CUTexture& CUTexture::operator=(CUTexture&& other) {
    valid_ = other.valid_;
    object_ = other.object_;
    other.valid_ = false;
    return *this;
}

CUTexture::operator cudaCUTexture_t() const {
    if (valid_) {
        return object_;
    } else {
        return cudaCUTexture_t{};
    }
}

}  // namespace cuda
}  // namespace decord