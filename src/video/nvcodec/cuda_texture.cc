/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_texture.cc
 * \brief NVCUVID texture objects
 */

#include "cuda_texture.h"
#include "nvcuvid/nvcuvid.h"

#include "../../runtime/cuda/cuda_common.h"

namespace decord {
namespace cuda {
using namespace runtime;

CUTexture::CUTexture() : valid_{false} {
}

CUTexture::CUTexture(const cudaResourceDesc* pResDesc,
                     const cudaTextureDesc* pTexDesc,
                     const cudaResourceViewDesc* pResViewDesc)
    : valid_{false}
{
    if (!CHECK_CUDA_CALL(cudaCreateTextureObject(&object_, pResDesc, pTexDesc, pResViewDesc))) {
        LOG(FATAL) << "Unable to create a texture object";
    }
    valid_ = true;
}

CUTexture::~CUTexture() {
    if (valid_) {
        cudaDestroyTextureObject(object_);
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

CUTexture::operator cudaTextureObject_t() const {
    if (valid_) {
        return object_;
    } else {
        return cudaTextureObject_t{};
    }
}

CUTextureRegistry::CUTextureRegistry() {

}

const CUImageTexture& CUTextureRegistry::GetTexture(uint8_t* ptr, unsigned int input_pitch,
                                                    uint16_t input_width, uint16_t input_height,
                                                    ScaleMethod scale_method, ChromaUpMethod chroma_up_method) {
    auto tex_id = std::make_tuple(ptr, scale_method, chroma_up_method);

    // find existing registed texture, if so return directly
    auto tex = textures_.find(tex_id);
    if (tex != textures_.end()) {
        return tex->second;
    }

    // not found, create new texture object
    CUImageTexture tex_object;
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    if (scale_method == ScaleMethod_Nearest) {
        tex_desc.filterMode   = cudaFilterModePoint;
    } else {
        tex_desc.filterMode   = cudaFilterModeLinear;
    }
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 0;

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = ptr;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
    res_desc.res.pitch2D.width = input_width;
    res_desc.res.pitch2D.height = input_height;
    res_desc.res.pitch2D.pitchInBytes = input_pitch;

    tex_object.luma = CUTexture{&res_desc, &tex_desc, nullptr};

    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    // only one ChromaUpMethod for now...
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 0;

    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = ptr + (input_height * input_pitch);
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    res_desc.res.pitch2D.width = input_width / 2;  // YUV420
    res_desc.res.pitch2D.height = input_height / 2;  // YUV420
    res_desc.res.pitch2D.pitchInBytes = input_pitch;

    tex_object.chroma = CUTexture{&res_desc, &tex_desc, nullptr};

    auto p = textures_.emplace(tex_id, std::move(tex_object));
    if (!p.second) {
        LOG(FATAL) << "Unable to cache a new texture object.";
    }
    return p.first->second;
}

}  // namespace cuda
}  // namespace decord
