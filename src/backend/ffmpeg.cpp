/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.cpp
 * \brief FFMPEG implementations
 */

#include "ffmpeg.h"
#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGVideoReader::FFMPEGVideoReader(std::string& fn, uint32_t vstream_idx) : ptr_fmt_context_(nullptr) {
    // allocate format context
    ptr_fmt_context_ = avformat_alloc_context();
    if (!ptr_fmt_context_) {
        LOG(FATAL) << "ERROR allocating memory for Format Context";
    }
    // open file
    if(avformat_open_input(&ptr_fmt_context_, fn.c_str(), NULL, NULL) != 0 ) {
        LOG(FATAL) << "ERROR opening file: " << fn;
    }

    // find stream info
    if (avformat_find_stream_info(ptr_fmt_context_,  NULL) < 0) {
        LOG(FATAL) << "ERROR getting stream info of file" << fn;
    }
    
    // iterate streams and set video stream index properly
    // note that vstream_idx does not count for audio streams
    // say this video has 4 streams (0:a, 1:v, 2:a, 3:v) 
    // with vstream_idx = 1(the second video stream), the real stream selected is 3:v

}

}  // namespace ffmpeg
}  // namespace decord