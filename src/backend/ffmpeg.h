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
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#ifdef __cplusplus
}
#endif


namespace decord {
namespace ffmpeg {
class FFMPEGVideoReader : public VideoReader {
  public:
      FFMPEGVideoReader(std::string& fn);
      ~FFMPEGVideoReader();
      void SetVideoStream(int stream_nb = -1);
      unsigned int QueryStreams();
  private:
      /*! \brief Video Streams Codecs in original videos */
      std::vector<AVCodec*> codecs_;
      /*! \brief Currently active video stream index */
      int actv_stm_idx_;
      /*! \brief AV format context holder */
      AVFormatContext *fmt_ctx_;
      /*! \brief AV dodec context for decoding related info */
      AVCodecContext *dec_ctx_;

};  // class FFMPEGVideoReader
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_