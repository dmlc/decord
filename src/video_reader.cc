/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader implementations
 */
#include <dmlc/logging.h>
#include <decord/video_reader.h>
#include "decoder/ffmpeg.h"


namespace decord {
VideoReaderPtr GetVideoReader(std::string fn, Decoder be) {
    std::shared_ptr<VideoReaderInterface> ptr;
    if (be == Decoder::FFMPEG()) {
        ptr = std::shared_ptr<VideoReaderInterface>(new ffmpeg::FFMPEGVideoReader(fn));
    } else {
        LOG(FATAL) << "Not supported Decoder type " << be;
    }
    return ptr;
}
}  // namespace decord