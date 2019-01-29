/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader implementations
 */

#include <decord/video_reader.h>
#include "backend/ffmpeg.h"
#include <dmlc/logging.h>

namespace decord {
std::shared_ptr<VideoReader> GetVideoReader(std::string& fn, Backend be = Backend::FFMPEG()) {
    std::shared_ptr<VideoReader> ptr;
    if (be == Backend::FFMPEG()) {
        ptr = std::make_shared<ffmpeg::FFMPEGVideoReader>(fn);
    } else {
        LOG(FATAL) << "Not supported backend type " << be;
    }
    return ptr;
}
}  // namespace decord