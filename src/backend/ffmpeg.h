/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.h
 * \brief FFMPEG related definitions
 */

#ifndef DECORD_BACKEND_FFMPEG_H_
#define DECORD_BACKEND_FFMPEG_H_

#include <decord/video_reader.h>
#include <decord/video_stream.h>
#include <string>
#include <vector>

namespace decord {
namespace ffmpeg {
class FFMPEGVideoReader : public VideoReader {
 public:
    FFMPEGVideoReader(std::string& fn);
    void SetVideoStream(uint32_t idx = 0);
 private:
    /*! \brief Video Streams in original videos */
    std::vector<VideoStream> vstreams_;
    /*! \brief Current selected stream index */
    uint32_t stream_idx_;

};  // class FFMPEGVideoReader
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_