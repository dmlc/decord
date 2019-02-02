/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader
 */

#ifndef DECORD_VIDEO_READER_H_
#define DECORD_VIDEO_READER_H_

#include <string>
#include <memory>

#include "base.h"
#include "ndarray.h"

namespace decord {

// Video Reader is an abtract class defining the reader interfaces
class VideoReader {
 public:
    /*! \brief query stream info in video */
    virtual unsigned int QueryStreams() = 0;
    /*! \brief check if video file successfully opened */
    virtual void SetVideoStream(int stream_nb = -1) = 0;
    /*! \brief read the next frame, return true if successful, false otherwise
    * Note that if NDArray is valid and has same shape, the memory will be reused
    * If NULL pointer is used, new NDArray will be created
    * If NDArray shape does not match desired frame shape, it will raise Error
    * */
    virtual bool NextFrame(NDArray* arr) = 0;
    /*! \brief destructor */
    virtual ~VideoReader() = default;
};  // class VideoReader

std::shared_ptr<VideoReader> GetVideoReader(std::string& fn, Decoder dec = Decoder::FFMPEG());
}  // namespace decord
#endif // DECORD_VIDEO_READER_H_