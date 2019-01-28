/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader
 */

#ifndef DECORD_VIDEO_READER_H_
#define DECORD_VIDEO_READER_H_

#include "base.h"
#include "video_stream.h"
#include <string>
#include <memory>

namespace decord {

// Video Reader is an abtract class defining the reader interfaces
class VideoReader {
 public:
    /*! \brief check if video file successfully opened */
    virtual void SetVideoStream(uint32_t idx = 0) = 0;
    /*! \brief destructor */
    virtual ~VideoReader() = default;
};  // class VideoReader

std::shared_ptr<VideoReader> GetVideoReader(std::string& fn, Backend be = Backend::FFMPEG());
}  // namespace decord
#endif // DECORD_VIDEO_READER_H_