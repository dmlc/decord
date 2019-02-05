/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.h
 * \brief FFMPEG related definitions
 */

#ifndef DECORD_BACKEND_FFMPEG_H_
#define DECORD_BACKEND_FFMPEG_H_

#include <decord/video_reader.h>

#include <string>
#include <vector>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
}
#endif
#include <decord/base.h>


namespace decord {
namespace ffmpeg {

/*! \brief FrameTransform as map key */
struct FrameTransform {
    AVPixelFormat fmt;
    uint32_t height;
    uint32_t width;
    uint32_t channel;
    int interp;  // interpolation method
    explicit FrameTransform(DLDataType dtype, uint32_t h, uint32_t w, uint32_t c, int interp);
};  // struct FrameTransform


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
          for (auto& sws_ctx : sws_ctx_map_) {
              sws_freeContext(sws_ctx.second);
          }
          sws_ctx_map_.clear();
      }
      void SetVideoStream(int stream_nb = -1);
      unsigned int QueryStreams();
      bool NextFrame(NDArray* arr);
  private:
      /*! \brief Get or Create SwsContext by dtype */
      struct SwsContext* GetSwsContext(FrameTransform out_fmt);

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
      /*! \brief Container for various FFMPEG swsContext */
      std::unordered_map<FrameTransform, struct SwsContext*> sws_ctx_map_;

};  // class FFMPEGVideoReader


}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_