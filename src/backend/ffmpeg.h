/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.h
 * \brief FFMPEG related definitions
 */

#ifndef DECORD_BACKEND_FFMPEG_H_
#define DECORD_BACKEND_FFMPEG_H_

#include <decord/video_reader.h>
#include <string>

namespace decord {
namespace ffmpeg {
class FFMPEGVideoReader : public VideoReader {
 public:
    FFMPEGVideoReader(std::string& fn);

};  // class FFMPEGVideoReader
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_