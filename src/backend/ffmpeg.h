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
      FFMPEGVideoReader(std::string& fn);
      void SetVideoStream(uint32_t idx = 0);
      size_t QueryVideoStreams();
  private:
      /*! \brief Video Streams Codecs in original videos */
      std::vector<std::pair<AVStream*, AVCodec*> > vstreams_;
      /*! \brief Currently selected video stream index */
      AVStream* active_vstream_;
      /*! \brief AV format context holder */
      AVFormatContext *ptr_fmt_ctx_;
      AVCodecContext *ptr_codec_ctx_;

};  // class FFMPEGVideoReader
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_