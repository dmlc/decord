/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cu_mapped_frame.cc
 * \brief NVCUVID mapped frame
 */

#include "cu_mapped_frame.h"
#include "cu_utils.h"

namespace decord {
namespace cuda {

CUMappedFrame::CUMappedFrame()
    : disp_info{nullptr}, valid_{false} {
}

CUMappedFrame::CUMappedFrame(CUVIDPARSERDISPINFO* disp_info,
                                    CUvideodecoder decoder,
                                    CUstream stream)
    : disp_info{disp_info}, valid_{false}, decoder_(decoder), params_{0} {

    if (!disp_info->progressive_frame) {
        LOG(FATAL) << "Got an interlaced frame. We don't do interlaced frames.";
    }

    params_.progressive_frame = disp_info->progressive_frame;
    params_.top_field_first = disp_info->top_field_first;
    params_.second_field = 0;
    params_.output_stream = stream;

    if (!CUDA_CHECK_CALL(cuvidMapVideoFrame(decoder_, disp_info->picture_index,
                                   &ptr_, &pitch_, &params_))) {
        LOG(FATAL) << "Unable to map video frame";
    }
    valid_ = true;
}

CUMappedFrame::CUMappedFrame(CUMappedFrame&& other)
    : disp_info{other.disp_info}, valid_{other.valid_}, decoder_{other.decoder_},
      ptr_{other.ptr_}, pitch_{other.pitch_}, params_{other.params_} {
    other.disp_info = nullptr;
    other.valid_ = false;
}

CUMappedFrame::~CUMappedFrame() {
    if (valid_) {
        if (!CUDA_CHECK_CALL(cuvidUnmapVideoFrame(decoder_, ptr_))) {
            LOG(FATAL) << "Error unmapping video frame";
        }
    }
}

uint8_t* CUMappedFrame::get_ptr() const {
    return reinterpret_cast<uint8_t*>(ptr_);
}

unsigned int CUMappedFrame::get_pitch() const {
    return pitch_;
}

}  // namespace cuda
}  // namespace decord
