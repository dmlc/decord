/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffmpeg.cpp
 * \brief FFMPEG implementations
 */

#include "ffmpeg.h"
#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGVideoReader::FFMPEGVideoReader(std::string& fn)
     : ptr_fmt_ctx_(NULL), ptr_codec_ctx_(NULL) {
    // allocate format context
    ptr_fmt_ctx_ = avformat_alloc_context();
    if (!ptr_fmt_ctx_) {
        LOG(FATAL) << "ERROR allocating memory for Format Context";
    }
    // open file
    if(avformat_open_input(&ptr_fmt_ctx_, fn.c_str(), NULL, NULL) != 0 ) {
        LOG(FATAL) << "ERROR opening file: " << fn;
    }

    // find stream info
    if (avformat_find_stream_info(ptr_fmt_ctx_,  NULL) < 0) {
        LOG(FATAL) << "ERROR getting stream info of file" << fn;
    }

    // initialize all video streams and store codecs info
    bool vstream_set = false;
    vstreams_.clear();
    for (uint32_t i = 0; i < ptr_fmt_ctx_->nb_streams; ++i) {
        AVStream *st = ptr_fmt_ctx_->streams[i];
        // LOG(INFO) << "AVStream->avg_frame_rate: " 
        //     << st->avg_frame_rate.num
        //     << "/" << st->avg_frame_rate.den;
        AVCodec *local_codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // store video stream codecs only
            vstreams_.emplace_back(std::make_pair(st, local_codec));
            if (!vstream_set) {
                SetVideoStream(i);
                vstream_set = true;
            }
        }
    }
    CHECK(vstream_set) << "No active video stream available!";
}

void FFMPEGVideoReader::SetVideoStream(uint32_t vstream_idx) {
    // iterate streams and set video stream index properly
    // note that vstream_idx does not count for audio streams
    // say this video has 4 streams (0:a, 1:v, 2:a, 3:v) 
    // with vstream_idx = 1(the second video stream), the real stream selected is 3:v
    CHECK(ptr_fmt_ctx_ != NULL);
    CHECK_LT(vstream_idx, vstreams_.size()) 
        << "vstream_idx " << vstream_idx << " out of bound " << vstreams_.size();
    CHECK(vstreams_[vstream_idx].second != NULL) 
        << "Video Stream [" << vstream_idx << "] format not supported";
    active_vstream_ = vstreams_[vstream_idx].first;
    // initialize the mem for codec context
    ptr_codec_ctx_ = avcodec_alloc_context3(vstreams_[vstream_idx].second);
    CHECK(ptr_codec_ctx_) << "ERROR allocating memory for AVCodecContext";
    // copy codec parameters to context
    CHECK_GE(avcodec_parameters_to_context(ptr_codec_ctx_, active_vstream_->codecpar), 0)
        << "ERROR copying codec parameters to context";
    // initialize AVCodecContext to use given AVCodec
    CHECK_GE(avcodec_open2(ptr_codec_ctx_, vstreams_[vstream_idx].second, NULL), 0)
        << "ERROR open codec through avcodec_open2";
}

size_t FFMPEGVideoReader::QueryVideoStreams() {
    CHECK(ptr_fmt_ctx_ != NULL);
    for (size_t i = 0; i < vstreams_.size(); ++i) {
        // iterate and print video stream info
        // feel free to add more if needed
        AVStream *st = vstreams_[i].first;
        AVCodec *codec = vstreams_[i].second;
        LOG(INFO) << "Video Stream [" << i << "]:"
        << " Average FPS: " 
        << static_cast<float>(st->avg_frame_rate.num) / st->avg_frame_rate.den
        << " Start time: "
        << st->start_time
        << " Duration: "
        << st->duration
        << " Codec Type: "
        << codec->name
        << " ID: "
        << codec->id
        << " bit_rate: "
        << st->codecpar->bit_rate
        << " Resolution: "
        << st->codecpar->width << "x" << st->codecpar->height;
    }
    return vstreams_.size();
}

}  // namespace ffmpeg
}  // namespace decord