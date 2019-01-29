/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.h
 * \brief FFMPEG related definitions
 */

#ifndef DECORD_BACKEND_FFMPEG_H_
#define DECORD_BACKEND_FFMPEG_H_

#include <decord/video_reader.h>
#include <decord/video_stream.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <string>
#include <vector>

namespace decord {
namespace ffmpeg {
class FFMPEGVideoReader : public VideoReader {
  public:
      FFMPEGVideoReader(std::string& fn, uint32_t vstream_idx = 0);
      void SetVideoStream(uint32_t idx = 0);
      size_t GetNumVideoStream();
  private:
      /*! \brief Video Streams in original videos */
      std::vector<VideoStream> vstreams_;
      /*! \brief Currently selected video stream index */
      uint32_t vstream_idx_;
      /*! \brief AV format context holder */
      AVFormatContext *ptr_fmt_context_;

};  // class FFMPEGVideoReader
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_