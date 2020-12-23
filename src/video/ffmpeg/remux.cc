/*!
 *  Copyright (c) 2020 by Contributors if not otherwise specified
 * \file remux.cc
 * \brief FFmpeg online remux implementation
 */

#include "remux.h"

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

int InMemoryRemux(ffmpeg::AVFormatContextPtr fmt_ctx, std::string& out_filename) {
    AVFormatContext *ifmt_ctx = fmt_ctx.get(), *ofmt_ctx = NULL;
    AVOutputFormat *ofmt = NULL;
    AVPacket pkt;
    int ret, i;
    av_dump_format(ifmt_ctx, 0, "", 0);
    out_filename = "/tmp/test_remux.de"
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, out_filename.c_str();
    if (!ofmt_ctx) {
        LOG(WARNING) << "Could not create output context";
        ret = AVERROR_UNKNOWN;
    }
    for (i = 0; i < ifmt_ctx->nb_streams; i++) {
        AVStream *in_stream = ifmt_ctx->streams[i];
        AVStream *out_stream = avformat_new_stream(ofmt_ctx, in_stream->codec->codec);
        if (!out_stream) {
            LOG(WARNING) << "Failed allocating output stream";
            ret = AVERROR_UNKNOWN;
        }

        ret = avcodec_copy_context(out_stream->codec, in_stream->codec);
        if (ret < 0) {
            LOG(WARNING) << "Failed to copy context from input to output stream codec context";
        }
        out_stream->codec->codec_tag = 0;
        if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
            out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      }
      av_dump_format(ofmt_ctx, 0, out_filename.c_str(), 1);

      if (!(ofmt->flags & AVFMT_NOFILE)) {
          ret = avio_open(&ofmt_ctx->pb, out_filename.c_str(), AVIO_FLAG_WRITE);
      if (ret < 0) {
          LOG(WARNING) << "Could not open output file " << out_filename;
      }
      }

      ret = avformat_write_header(ofmt_ctx, NULL);
      if (ret < 0) {
          LOG(WARNING) << "Error occurred when opening output file";
      }

      while (1) {
          AVStream *in_stream, *out_stream;

          ret = av_read_frame(ifmt_ctx, &pkt);
          if (ret < 0)
              break;

          in_stream  = ifmt_ctx->streams[pkt.stream_index];
          out_stream = ofmt_ctx->streams[pkt.stream_index];

          // log_packet(ifmt_ctx, &pkt, "in");

          /* copy packet */
          pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
          pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
          pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
          pkt.pos = -1;
          // log_packet(ofmt_ctx, &pkt, "out");

          ret = av_interleaved_write_frame(ofmt_ctx, &pkt);
          if (ret < 0) {
              LOG(WARNING) << "Error muxing packet";
              break;
          }
          av_free_packet(&pkt);
      }

      av_write_trailer(ofmt_ctx);

      if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE)) {
          avio_closep(&ofmt_ctx->pb);
      }
      avformat_free_context(ofmt_ctx);

      if (ret < 0 && ret != AVERROR_EOF) {
        LOG(WARNING) << "Error occurred: " << av_err2str(ret);
        return 1;
      }

    return 0;
}

}  // namespace ffmpeg
}  // namespace decord
