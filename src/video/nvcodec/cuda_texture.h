/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_texture.h
 * \brief NVCUVID texture objects
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_TEXTURE_H_
#define DECORD_VIDEO_NVCODEC_CUDA_TEXTURE_H_

#include "nvcuvid/nvcuvid.h"
#include "../../runtime/cuda/cuda_common.h"

#include <unordered_map>

#include <dmlc/base.h>

namespace decord {
namespace cuda {

/**
 * How the image is scaled up/down from the original
 */
enum ScaleMethod {
    /**
     * The value for the nearest neighbor is used, no interpolation
     */
    ScaleMethod_Nearest,

    /**
     * Simple bilinear interpolation of four nearest neighbors
     */
    ScaleMethod_Linear

    // These are possibilities but currently unimplemented (PRs welcome)
    // ScaleMethod_Area
    // ScaleMethod_Cubic
    // ScaleMethod_Lanczos
};

/**
 * How the chroma channels are upscaled from yuv 4:2:0 to 4:4:4
 */
enum ChromaUpMethod {
    /**
     * Simple bilinear interpolation of four nearest neighbors
     */
    ChromaUpMethod_Linear

    // These are possibilities but currently unimplemented (PRs welcome)
    // ChromaUpMethod_CatmullRom
};

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


/**
 * \brief A registry of CUDA Texture Objects, for fast retrieval
 *
 */
class CUTextureRegistry {
    public:
        // Here we assume that a data pointer, scale method, chroma_up_method uniquely defines a texture
        using TexID = std::tuple<uint8_t*, ScaleMethod, ChromaUpMethod>;
        CUTextureRegistry();
        const CUImageTexture& GetTexture(uint8_t* ptr, unsigned int input_pitch,
                                         uint16_t input_width, uint16_t input_height,
                                         ScaleMethod scale_method, ChromaUpMethod chroma_up_method);

    private:
        struct TexHash {
        std::hash<uint8_t*> ptr_hash;
        std::hash<int> scale_hash;
        std::hash<int> up_hash;
        std::size_t operator () (const TexID& tex) const {
            return ptr_hash(std::get<0>(tex))
                    ^ scale_hash(std::get<1>(tex))
                    ^ up_hash(std::get<2>(tex));
            }
        };  // struct TexHash
        std::unordered_map<TexID, CUImageTexture, TexHash> textures_;

};  // class CUTextureRegistry

}  // namespace cuda
}  // namespace decord

#endif
