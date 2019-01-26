/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_stream.h
 * \brief Video Stream container
 */

#ifndef DECORD_VIDEO_STREAM_H_
#define DECORD_VIDEO_STREAM_H_

namespace decord {
// VideoStream is an abstract class defining interfaces for video streams
class VideoStream {
 public:
    /*!
    * \brief Destructor.
    */
    virtual ~VideoStream() = default;
};  // class VideoStream

}  // namespace decord

#endif  // DECORD_VIDEO_STREAM_H_