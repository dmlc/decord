/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader implementations
 */

#include <decord/video_reader.h>
#include "backend/ffmpeg.h"

namespace decord {
std::shared_ptr<VideoReader> GetVideoReader(std::string& fn, Backend be = Backend::FFMPEG()) {
    if (be == Backend::FFMPEG()) {
        auto ptr = std::make_shared<ffmpeg::FFMPEGVideoReader>(fn);
    } else {
        LOG(FATAL) << "Not supported backend type " << be;
    }
}
}  // namespace decord