/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_interface.cc
 * \brief Video file reader implementations
 */

#include "./ffmpeg/video_reader.h"

#include <decord/video_interface.h>


#include <dmlc/logging.h>

namespace decord {

VideoReaderPtr GetVideoReader(std::string fn, Decoder be) {
    std::shared_ptr<VideoReaderInterface> ptr;
    if (be == Decoder::FFMPEG()) {
        // ptr = std::shared_ptr<VideoReaderInterface>(new ffmpeg::FFMPEGVideoReader(fn));
        ptr = std::make_shared<ffmpeg::FFMPEGVideoReader>(fn);
    } else {
        LOG(FATAL) << "Not supported Decoder type " << be;
    }
    return ptr;
}
}  // namespace decord