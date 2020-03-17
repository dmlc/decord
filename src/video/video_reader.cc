/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_reader.cc
 * \brief Video reader Impl
 */

#include "video_reader.h"
#include "ffmpeg/threaded_decoder.h"
#if DECORD_USE_CUDA
#include "nvcodec/cuda_threaded_decoder.h"
#endif
#include <algorithm>
#include <decord/runtime/ndarray.h>

namespace decord {

using NDArray = runtime::NDArray;
using AVFramePtr = ffmpeg::AVFramePtr;
using AVPacketPtr = ffmpeg::AVPacketPtr;
using FFMPEGThreadedDecoder = ffmpeg::FFMPEGThreadedDecoder;
using AVFramePool = ffmpeg::AVFramePool;
using AVPacketPool = ffmpeg::AVPacketPool;

VideoReader::VideoReader(std::string fn, DLContext ctx, int width, int height)
     : ctx_(ctx), key_indices_(), frame_ts_(), codecs_(), actv_stm_idx_(-1), decoder_(), curr_frame_(0),
     width_(width), height_(height), eof_(false) {
    // av_register_all deprecated in latest versions
    #if ( LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,9,100) )
    av_register_all();
    #endif

    AVFormatContext *fmt_ctx = nullptr;
    int open_ret = avformat_open_input(&fmt_ctx, fn.c_str(), NULL, NULL);
    if( open_ret != 0 ) {
        char errstr[200];
        av_strerror(open_ret, errstr, 200);
        LOG(FATAL) << "ERROR opening file: " << fn.c_str() << ", " << errstr;
        return;
    }
    fmt_ctx_.reset(fmt_ctx);

    // find stream info
    if (avformat_find_stream_info(fmt_ctx,  NULL) < 0) {
        LOG(FATAL) << "ERROR getting stream info of file" << fn;
    }

    // initialize all video streams and store codecs info
    for (uint32_t i = 0; i < fmt_ctx_->nb_streams; ++i) {
        AVStream *st = fmt_ctx_->streams[i];
        AVCodec *local_codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // store video stream codecs only
            codecs_.emplace_back(local_codec);
        } else {
            // audio, subtitle... skip
            AVCodec *tmp = NULL;
            codecs_.emplace_back(tmp);
        }
    }
    // LOG(INFO) << "initialized all streams";
    // find best video stream (-1 means auto, relay on FFMPEG)
    SetVideoStream(-1);
    // LOG(INFO) << "Set video stream";
    decoder_->Start();

    // // allocate AVFrame buffer
    // frame_ = av_frame_alloc();
    // CHECK(frame_) << "ERROR failed to allocated memory for AVFrame";

    // // allocate AVPacket buffer
    // pkt_ = av_packet_alloc();
    // CHECK(pkt_) << "ERROR failed to allocated memory for AVPacket";
}

VideoReader::~VideoReader(){
    // avformat_free_context(fmt_ctx_);
    // avformat_close_input(&fmt_ctx_);
    // LOG(INFO) << "Destruct Video REader";
}

void VideoReader::SetVideoStream(int stream_nb) {
    CHECK(fmt_ctx_ != NULL);
    AVCodec *dec;
    int st_nb = av_find_best_stream(fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, stream_nb, -1, &dec, 0);
    // LOG(INFO) << "find best stream: " << st_nb;
    CHECK_GE(st_nb, 0) << "ERROR cannot find video stream with wanted index: " << stream_nb;
    // initialize the mem for codec context
    CHECK(codecs_[st_nb] == dec) << "Codecs of " << st_nb << " is NULL";
    // LOG(INFO) << "codecs of stream: " << codecs_[st_nb] << " name: " <<  codecs_[st_nb]->name;
    ffmpeg::AVCodecParametersPtr codecpar;
    codecpar.reset(avcodec_parameters_alloc());
    CHECK_GE(avcodec_parameters_copy(codecpar.get(), fmt_ctx_->streams[st_nb]->codecpar), 0)
        << "Error copy stream->codecpar to buffer codecpar";
    if (kDLCPU == ctx_.device_type) {
        decoder_ = std::unique_ptr<ThreadedDecoderInterface>(new FFMPEGThreadedDecoder());
    } else if (kDLGPU == ctx_.device_type) {
#ifdef DECORD_USE_CUDA
        // note: cuda threaded decoder will modify codecpar
        decoder_ = std::unique_ptr<ThreadedDecoderInterface>(new cuda::CUThreadedDecoder(
            ctx_.device_id, codecpar.get()));
#else
        LOG(FATAL) << "CUDA not enabled. Requested context GPU(" << ctx_.device_id << ").";
#endif
    } else {
        LOG(FATAL) << "Unknown device type: " << ctx_.device_type;
    }

    auto dec_ctx = avcodec_alloc_context3(dec);
	dec_ctx->thread_count = 0;
	// LOG(INFO) << "Original decoder multithreading: " << dec_ctx->thread_count;
    // CHECK_GE(avcodec_copy_context(dec_ctx, fmt_ctx_->streams[stream_nb]->codec), 0) << "Error: copy context";
    // CHECK_GE(avcodec_parameters_to_context(dec_ctx, fmt_ctx_->streams[st_nb]->codecpar), 0) << "Error: copy parameters to codec context.";
    // copy codec parameters to context
    CHECK_GE(avcodec_parameters_to_context(dec_ctx, codecpar.get()), 0)
        << "ERROR copying codec parameters to context";
    // initialize AVCodecContext to use given AVCodec
    int open_ret = avcodec_open2(dec_ctx, codecs_[st_nb], NULL);
    if (open_ret < 0 ) {
        char errstr[200];
        av_strerror(open_ret, errstr, 200);
        LOG(FATAL) << "ERROR open codec through avcodec_open2: " << errstr;
        return;
    }
    // CHECK_GE(avcodec_open2(dec_ctx, codecs_[st_nb], NULL), 0)
    //     << "ERROR open codec through avcodec_open2";
    // LOG(INFO) << "codecs opened.";
    actv_stm_idx_ = st_nb;
    // LOG(INFO) << "time base: " << fmt_ctx_->streams[st_nb]->time_base.num << " / " << fmt_ctx_->streams[st_nb]->time_base.den;
    dec_ctx->time_base = fmt_ctx_->streams[st_nb]->time_base;
    if (width_ < 1) {
        width_ = codecpar->width;
    }
    if (height_ < 1) {
        height_ = codecpar->height;
    }

    ndarray_pool_ = NDArrayPool(32, {height_, width_, 3}, kUInt8, ctx_);
    decoder_->SetCodecContext(dec_ctx, width_, height_);
    IndexKeyframes();
}

unsigned int VideoReader::QueryStreams() const {
    CHECK(fmt_ctx_ != NULL);
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; ++i) {
        // iterate and print stream info
        // feel free to add more if needed
        AVStream *st = fmt_ctx_->streams[i];
        AVCodec *codec = codecs_[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            LOG(INFO) << "video stream [" << i << "]:"
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
                << st->codecpar->width << "x" << st->codecpar->height
                << " Frame count: "
                << st->nb_frames;
        } else {
            const char *codec_type = av_get_media_type_string(st->codecpar->codec_type);
            codec_type = codec_type? codec_type : "unknown type";
            LOG(INFO) << codec_type << " stream [" << i << "].";
        }

    }
    return fmt_ctx_->nb_streams;
}

int64_t VideoReader::GetFrameCount() const {
    if (frame_ts_.size() > 0) {
        return frame_ts_.size();
    }
    CHECK(fmt_ctx_ != NULL);
    CHECK(actv_stm_idx_ >= 0);
    CHECK(actv_stm_idx_ >= 0 && static_cast<unsigned int>(actv_stm_idx_) < fmt_ctx_->nb_streams);
    int64_t cnt = fmt_ctx_->streams[actv_stm_idx_]->nb_frames;
    if (cnt < 1) {
        AVStream *stm = fmt_ctx_->streams[actv_stm_idx_];
        // many formats do not provide accurate frame count, use duration and FPS to approximate
        cnt = static_cast<double>(stm->avg_frame_rate.num) / stm->avg_frame_rate.den * fmt_ctx_->duration / AV_TIME_BASE;
    }
    return cnt;
}

int64_t VideoReader::GetCurrentPosition() const {
    return curr_frame_;
}

int64_t VideoReader::FrameToPTS(int64_t pos) {
    int64_t ts = pos * fmt_ctx_->streams[actv_stm_idx_]->duration / GetFrameCount();
    return ts;
}

std::vector<int64_t> VideoReader::FramesToPTS(const std::vector<int64_t>& positions) {
    auto nframe = GetFrameCount();
    auto duration = fmt_ctx_->streams[actv_stm_idx_]->duration;
    std::vector<int64_t> ret;
    ret.reserve(positions.size());
    for (auto pos : positions) {
        ret.emplace_back(pos * duration / nframe);
    }
    return ret;
}

bool VideoReader::Seek(int64_t pos) {
    if (curr_frame_ == pos) return true;
    decoder_->Clear();
    eof_ = false;

    int64_t ts = FrameToPTS(pos);
    int ret = av_seek_frame(fmt_ctx_.get(), actv_stm_idx_, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) LOG(WARNING) << "Failed to seek file to position: " << pos;
    // LOG(INFO) << "seek return: " << ret;
    decoder_->Start();
    if (ret >= 0) {
        curr_frame_ = pos;
    }
    return ret >= 0;
}

int64_t VideoReader::LocateKeyframe(int64_t pos) {
    if (key_indices_.size() < 1) return 0;
    if (pos <= key_indices_[0]) return 0;
    if (pos >= GetFrameCount()) return key_indices_.back();
    auto it = std::upper_bound(key_indices_.begin(), key_indices_.end(), pos) - 1;
    return *it;
}

bool VideoReader::SeekAccurate(int64_t pos) {
    if (curr_frame_ == pos) return true;
    int64_t key_pos = LocateKeyframe(pos);
    int64_t curr_key_pos = LocateKeyframe(curr_frame_);
    if (key_pos != curr_key_pos) {
        // need to seek to keyframes first
        bool ret = Seek(key_pos);
        if (!ret) return false;
        SkipFrames(pos - key_pos);
    } else if (pos < curr_frame_) {
        // need seek backwards to the nearest keyframe
        bool ret = Seek(key_pos);
        if (!ret) return false;
        SkipFrames(pos - key_pos);
    } else {
        // no need to seek to keyframe, since both current and seek position belong to same keyframe
        SkipFrames(pos - curr_frame_);
    }
    return true;
}

void VideoReader::PushNext() {
    // AVPacket *packet = av_packet_alloc();
    AVPacketPtr packet = AVPacketPool::Get()->Acquire();
    int ret = -1;
    while (!eof_) {
        ret = av_read_frame(fmt_ctx_.get(), packet.get());
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                eof_ = true;
                // flush buffer
                if (ctx_.device_type != kDLGPU) {
                    // no preallocated memory and memory pool, use FFMPEG AVFrame pool
                    decoder_->Push(nullptr, NDArray());
                } else {
                    // use preallocated memory pool for GPU
                    decoder_->Push(nullptr, ndarray_pool_.Acquire());
                }
                return;
            } else {
                LOG(FATAL) << "Error: av_read_frame failed with " << AVERROR(ret);
            }
            return;
        }
        if (packet->stream_index == actv_stm_idx_) {

            if (ctx_.device_type != kDLGPU) {
                    // no preallocated memory and memory pool, use FFMPEG AVFrame pool
                    decoder_->Push(packet, NDArray());
                } else {
                    // use preallocated memory pool for GPU
                    decoder_->Push(packet, ndarray_pool_.Acquire());
                }
            // LOG(INFO) << "Pushed packet to decoder.";
            break;
        }
        av_packet_unref(packet.get());
    }
}

NDArray VideoReader::NextFrameImpl() {
    NDArray frame;
    decoder_->Start();
    bool ret = false;
    int rewind_offset = 0;
    while (!ret) {
        PushNext();
        if (curr_frame_ >= GetFrameCount()) {
            return NDArray::Empty({}, kUInt8, ctx_);
        }
        ret = decoder_->Pop(&frame);
        if (frame.Size() <= 1) {
            if (frame.defined() && frame.data_->dl_tensor.dtype == kInt64) {
                SeekAccurate(curr_frame_ - rewind_offset);
                ++rewind_offset;
            }
            ret = false;
        }
    }
    if (frame.defined()) {
        ++curr_frame_;
    }
    return frame;
}

NDArray VideoReader::NextFrame() {
    return  NextFrameImpl();
}

void VideoReader::IndexKeyframes() {
    CHECK(actv_stm_idx_ >= 0) << "Invalid active stream index, not yet initialized!";
    key_indices_.clear();
    frame_ts_.clear();
    AVPacketPtr packet = AVPacketPool::Get()->Acquire();
    int ret = -1;
    bool eof = false;
    int64_t cnt = 0;
    frame_ts_.reserve(GetFrameCount());
    timestamp_t start_sec = fmt_ctx_->streams[actv_stm_idx_]->start_time;
    auto stm_ts = fmt_ctx_->streams[actv_stm_idx_]->time_base;
    double ts_factor = stm_ts.den == 0 || stm_ts.num == 0 ? 0. : (double)stm_ts.num / (double)stm_ts.den;

    while (!eof) {
        ret = av_read_frame(fmt_ctx_.get(), packet.get());
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                eof = true;
                break;
            } else {
                LOG(FATAL) << "Error: av_read_frame failed with " << AVERROR(ret);
            }
            break;
        }
        if (packet->stream_index == actv_stm_idx_) {
            // store the pts info for each frame
            auto start_pts = (packet->pts - start_sec) * ts_factor;
            auto stop_pts = (packet->pts + packet->duration - start_sec) * ts_factor;
            frame_ts_.emplace_back(AVFrameTime(packet->pts, packet->dts, start_pts, stop_pts));
            if (packet->flags & AV_PKT_FLAG_KEY) {
                key_indices_.emplace_back(cnt);
            }
            ++cnt;
        }
        av_packet_unref(packet.get());
    }
    curr_frame_ = GetFrameCount();
	ret = Seek(0);
}

runtime::NDArray VideoReader::GetKeyIndices() {
    std::vector<int64_t> shape = {static_cast<int64_t>(key_indices_.size())};
    runtime::NDArray ret = runtime::NDArray::Empty(shape, kInt64, kCPU);
    ret.CopyFrom<int64_t>(key_indices_, shape);
    return ret;
}

runtime::NDArray VideoReader::GetFramePTS() const {
    // copy to ndarray
    std::vector<float> tmp(frame_ts_.size() * 2, 0);
    for (size_t i = 0; i < frame_ts_.size(); ++i) {
        auto pos = i << 1;
        tmp[pos] = frame_ts_[i].start;
        tmp[pos + 1] = frame_ts_[i].stop;
    }
    std::vector<int64_t> shape = {static_cast<int64_t>(frame_ts_.size()), 2};
    runtime::NDArray ret = runtime::NDArray::Empty(shape, kFloat32, kCPU);
    ret.CopyFrom<float>(tmp, shape);
    return ret;
}

double VideoReader::GetAverageFPS() const {
    CHECK(actv_stm_idx_ >= 0);
    CHECK(static_cast<unsigned int>(actv_stm_idx_) < fmt_ctx_->nb_streams);
    AVStream *active_st = fmt_ctx_->streams[actv_stm_idx_];
    return static_cast<double>(active_st->avg_frame_rate.num) / active_st->avg_frame_rate.den;
}

std::vector<int64_t> VideoReader::GetKeyIndicesVector() const {
    return key_indices_;
}

void VideoReader::SkipFrames(int64_t num) {
    // check if skip pass keyframes, if so, we can seek to latest keyframe first
    // LOG(INFO) << " Skip Frame start: " << num << " current frame: " << curr_frame_;
    if (num < 1) return;
    num = std::min(GetFrameCount() - curr_frame_, num);
    auto it1 = std::upper_bound(key_indices_.begin(), key_indices_.end(), curr_frame_) - 1;
    CHECK_GE(it1 - key_indices_.begin(), 0);
    auto it2 = std::upper_bound(key_indices_.begin(), key_indices_.end(), curr_frame_ + num) - 1;
    CHECK_GE(it2 - key_indices_.begin(), 0);
    // LOG(INFO) << "first: " << it1 - key_indices_.begin() << " second: " << it2 - key_indices_.begin() << ", " << *it1 << ", " << *it2;
    if (it2 > it1) {
        int64_t old_frame = curr_frame_;
        // LOG(INFO) << "Seek to frame: " << *it2;
        Seek(*it2);
        // LOG(INFO) << "current: " << curr_frame_ << ", adjust skip from " << num << " to " << num + old_frame - *it2;
        num += old_frame - *it2;
    }
    if (num < 1) return;

    // LOG(INFO) << "started skipping with: " << num;
    NDArray frame;
    decoder_->Start();
    bool ret = false;
    std::vector<int64_t> frame_pos(num);
    std::iota(frame_pos.begin(), frame_pos.end(), curr_frame_);
    auto pts = FramesToPTS(frame_pos);
    decoder_->SuggestDiscardPTS(pts);
    curr_frame_ += num;
    while (num >= 0) {
        PushNext();
        ret = decoder_->Pop(&frame);
        if (!ret) continue;
        // LOG(INFO) << "skip: " << num;
        --num;
    }
    // LOG(INFO) << " stopped skipframes: " << curr_frame_;
}

NDArray VideoReader::GetBatch(std::vector<int64_t> indices, NDArray buf) {
    std::size_t bs = indices.size();
    // find the first occurance of each index to avoid duplicate access
    std::unordered_map<int64_t, std::size_t> unique_indices;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        int64_t value = indices[i];
        auto it = unique_indices.find(value);
        if (it == unique_indices.end()) {
            // first apperance only
            unique_indices[value] = i;
        }
    }
    if (!buf.defined()) {
        buf = NDArray::Empty({static_cast<int64_t>(bs), height_, width_, 3}, kUInt8, ctx_);
    }
    // LOG(INFO) << height_ << " "  << width_ << " Buf size: " << bs << " total: " << bs * height_ * width_ * 3;
    int64_t frame_count = GetFrameCount();
    uint64_t offset = 0;
    std::vector<int64_t> frame_shape = {height_, width_, 3};
    for (std::size_t i = 0; i < indices.size(); ++i) {
        int64_t pos = indices[i];
        auto it = unique_indices.find(pos);
        if (it != unique_indices.end() && it->second != i) {
            // not the first occurance of frame, try to copy from buffer rather than load from video
            CHECK(i > it->second);
            CHECK(i > 0);
            uint64_t old_offset = offset / i * it->second;
            auto old_view = buf.CreateOffsetView(frame_shape, kUInt8, &old_offset);
            auto view = buf.CreateOffsetView(frame_shape, kUInt8, &offset);
            old_view.CopyTo(view); 
        }
        else {
            CHECK_LT(pos, frame_count);
            CHECK_GE(pos, 0);
            if (curr_frame_ == pos) {
                // no need to seek
            } else if (pos > curr_frame_) {
                // skip positive number of frames
                SkipFrames(pos - curr_frame_);
            } else {
                // seek no matter what
                SeekAccurate(pos);
            }
            NDArray frame = NextFrameImpl();

            if (frame.Size() < 1 && eof_) {
                LOG(FATAL) << "Error getting frame at: " << pos << " with total frames: " << frame_count;
            }
            // copy frame to buffer
            // LOG(INFO) << "index: " << i << ", size: " << height_ * width_ * 3 * i <<  ", offset: " << offset << " Curr frame: " << frame.data_->dl_tensor.shape[0] << " x " << frame.data_->dl_tensor.shape[1] << " x " << frame.data_->dl_tensor.shape[2] << " Frame size: " << frame.Size();
            auto view = buf.CreateOffsetView(frame_shape, frame.data_->dl_tensor.dtype, &offset);
            frame.CopyTo(view);
        }
    }
    return buf;
}

}  // namespace decord
