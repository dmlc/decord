/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.h
 * \brief FFMPEG related definitions
 */

#ifndef DECORD_BACKEND_FFMPEG_H_
#define DECORD_BACKEND_FFMPEG_H_

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
#include <decord/video_reader.h>


namespace decord {
namespace ffmpeg {
class FFMPEGVideoReader : public VideoReader {
  public:
      FFMPEGVideoReader(std::string& fn);
      /*! \brief Destructor, note that FFMPEG resources has to be managed manually to avoid resource leak */
      ~FFMPEGVideoReader() {
          avformat_close_input(&fmt_ctx_);
          avformat_free_context(fmt_ctx_);
          av_packet_free(&pkt_);
          av_frame_free(&frame_);
          avcodec_free_context(&dec_ctx_);
      }
      void SetVideoStream(int stream_nb = -1);
      unsigned int QueryStreams();
      bool NextFrame(NDArray* arr, DLDataType dtype);
  private:
      /*! \brief Video Streams Codecs in original videos */
      std::vector<AVCodec*> codecs_;
      /*! \brief Currently active video stream index */
      int actv_stm_idx_;
      /*! \brief AV format context holder */
      AVFormatContext *fmt_ctx_;
      /*! \brief AVPacket buffer */
      AVPacket *pkt_;
      /*! \brief AVFrame buffer */
      AVFrame *frame_;
      /*! \brief AV dodec context for decoding related info */
      AVCodecContext *dec_ctx_;

};  // class FFMPEGVideoReader

/*! \brief Convert from raw AVFrame to NDArray with type */
bool ToNDArray(AVFrame *frame, NDArray *arr, DLDataType dtype);
}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_