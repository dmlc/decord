/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader implementations
 */
#include <dmlc/logging.h>
#include <decord/video_reader.h>
#include "decoder/ffmpeg.h"


namespace decord {
std::shared_ptr<VideoReaderInterface> GetVideoReader(std::string fn, Decoder be = Decoder::FFMPEG()) {
    std::shared_ptr<VideoReaderInterface> ptr;
    if (be == Decoder::FFMPEG()) {
        ptr = std::make_shared<ffmpeg::FFMPEGVideoReader>(fn);
    } else {
        LOG(FATAL) << "Not supported Decoder type " << be;
    }
    return ptr;
}
}  // namespace decord