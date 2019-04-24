/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_mapped_frame.h
 * \brief NVCUVID mapped frame
 */

#ifndef DECORD_VIDEO_NVCODEC_CU_MAPPED_FRAME_H_
#define DECORD_VIDEO_NVCODEC_CU_MAPPED_FRAME_H_

#include "nvcuvid/nvcuvid.h"

#include <dmlc/base.h>

namespace decord {
namespace cuda {

class CUMappedFrame {
    public:
        CUMappedFrame();
        CUMappedFrame(CUVIDPARSERDISPINFO* disp_info, CUvideodecoder decoder,
                    CUstream stream);
        ~CUMappedFrame();

        uint8_t* get_ptr() const;
        unsigned int get_pitch() const;

        CUVIDPARSERDISPINFO* disp_info;
    
    private:
        bool valid_;
        CUvideodecoder decoder_;
        CUdeviceptr ptr_;
        unsigned int pitch_;
        CUVIDPROCPARAMS params_;

    DISALLOW_COPY_AND_ASSIGN(CUMappedFrame);
};

}  // namespace cuda
}  // namespace decord

#endif