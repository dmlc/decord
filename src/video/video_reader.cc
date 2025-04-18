/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_reader.cc
 * \brief Video reader Impl
 */

#include "video_reader.h"
#include "ffmpeg/threaded_decoder.h"
#include "../runtime/str_util.h"
#if DECORD_USE_CUDA
#include "nvcodec/cuda_threaded_decoder.h"
#endif
#include <algorithm>
#include <decord/runtime/ndarray.h>
#include <decord/runtime/c_runtime_api.h>

namespace decord {

using NDArray = runtime::NDArray;
using AVFramePtr = ffmpeg::AVFramePtr;
using AVPacketPtr = ffmpeg::AVPacketPtr;
using FFMPEGThreadedDecoder = ffmpeg::FFMPEGThreadedDecoder;
using AVFramePool = ffmpeg::AVFramePool;
using AVPacketPool = ffmpeg::AVPacketPool;
// AVIO buffer size when reading from raw bytes
static const int AVIO_BUFFER_SIZE = std::stoi(runtime::GetEnvironmentVariableOrDefault("DECORD_AVIO_BUFFER_SIZE", "40960"));
// (corrupted video only): Max retry when cache frame is unavailable and rewind is required to decode a frame
static const int REWIND_RETRY_MAX = std::stoi(runtime::GetEnvironmentVariableOrDefault("DECORD_REWIND_RETRY_MAX", "16"));
// (corrupted video only): Max retry when eof is detected but last few frames are not available
static const int EOF_RETRY_MAX = std::stoi(runtime::GetEnvironmentVariableOrDefault("DECORD_EOF_RETRY_MAX", "10240"));
// (corrupted video only): The warning threshold(0.0 - 1.0) when multiple frames are unavailable and fallbacked to cached frames
static const float DUPLICATE_WARNING_THRESHOLD = std::stof(runtime::GetEnvironmentVariableOrDefault("DECORD_DUPLICATE_WARNING_THRESHOLD", "0.25"));

VideoReader::VideoReader(std::string fn, DLContext ctx, int width, int height, int nb_thread, int io_type, std::string fault_tol)
     : ctx_(ctx), key_indices_(), pts_frame_map_(), tmp_key_frame_(), overrun_(false), frame_ts_(), codecs_(),
     actv_stm_idx_(-1), fmt_ctx_(nullptr), decoder_(nullptr), curr_frame_(0),
     nb_thread_decoding_(nb_thread), width_(width), height_(height), eof_(false), io_ctx_(),
     use_cached_frame_(true), fault_tol_thresh_(-1.0), fault_warn_emit_(false) {
    // av_register_all deprecated in latest versions
    #if ( LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,9,100) )
    av_register_all();
    #endif

    AVFormatContext *fmt_ctx = nullptr;
    int open_ret = 1;
    if (io_type == kDevice) {
        LOG(WARNING) << "Not implemented";
        return;
        // #ifdef DECORD_USE_LIBAVDEVICE
        //     avdevice_register_all();
        //     fmt_ctx = avformat_alloc_context();
        //     CHECK(fmt_ctx) << "Unable to alloc avformat context";
        //     AVInputFormat *ifmt = av_find_input_format(io_format);
        //     CHECK(ifmt) << "Unable to find input format: " << io_format;
        //     std::string device_name = "video=" + fn;
        //     open_ret = avformat_open_input(&fmt_ctx, device_name.c_str(), ifmt, NULL);
        // #else
        //     LOG(WARNING) << "Unable to process device IO as decord is not built with libavdevice!";
        //     return;
        // #endif
    } else if (io_type == kRawBytes) {
        filename_ = "BytesIO";
        io_ctx_.reset(new ffmpeg::AVIOBytesContext(fn, AVIO_BUFFER_SIZE));
        fmt_ctx = avformat_alloc_context();
        CHECK(fmt_ctx != nullptr) << "Unable to alloc avformat context";
        fmt_ctx->pb = io_ctx_->get_avio();
        if (!fmt_ctx->pb) {
            LOG(WARNING) << "Unable to init AVIO from memory buffer";
            return;
        }
        open_ret = avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
    } else if (io_type == kNormal) {
        filename_ = fn;
        open_ret = avformat_open_input(&fmt_ctx, fn.c_str(), NULL, NULL);
    } else {
        LOG(WARNING) << "Invalid io type: " << io_type;
    }

    if( open_ret != 0 ) {
        char errstr[200];
        av_strerror(open_ret, errstr, 200);
        if (io_type != kBytes) {
            LOG(WARNING) << "ERROR opening: " << fn.c_str() << ", " << errstr;
        } else {
            LOG(WARNING) << "ERROR opening " << fn.size() << " bytes, " << errstr;
        }
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
        const AVCodec *local_codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // store video stream codecs only
            codecs_.emplace_back(local_codec);
        } else {
            // audio, subtitle... skip
            const AVCodec *tmp = NULL;
            codecs_.emplace_back(tmp);
        }
    }
    // LOG(INFO) << "initialized all streams";
    // find best video stream (-1 means auto, relay on FFMPEG)
    SetVideoStream(-1);
    // LOG(INFO) << "Set video stream";

    // init fault tolerance threshold
    if (fault_tol.size()) {
        int64_t absolute_thresh = -1;
        double percent_thresh = -1.0;
        int ft_ret = runtime::ParseIntOrFloat(fault_tol, absolute_thresh, percent_thresh);
        if (ft_ret == 0 && absolute_thresh) {
            // absolute, integer based risk control
            fault_tol_thresh_ = absolute_thresh;
        } else if (ft_ret == 1 && (percent_thresh > 0)) {
            // percentage
            fault_tol_thresh_ = percent_thresh * GetFrameCount();
        }
    }

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
    if (!fmt_ctx_) return;
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
            ctx_.device_id, codecpar.get(), fmt_ctx_->iformat));
#else
        LOG(FATAL) << "CUDA not enabled. Requested context GPU(" << ctx_.device_id << ").";
#endif
    } else {
        LOG(FATAL) << "Unknown device type: " << ctx_.device_type;
    }

    auto dec_ctx = avcodec_alloc_context3(dec);
    // LOG(INFO) << "nb_thread_decoding_: " << nb_thread_decoding_;
    dec_ctx->thread_count = nb_thread_decoding_;
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

    int rotation = static_cast<int>(GetRotation());
    int original_width = codecpar->width;
    int original_height = codecpar->height;

    if ((rotation == 90 || rotation == 270) && ctx_.device_type != kDLGPU) {
        std::swap(original_width, original_height);
    }

    if (width_ < 1) {
        width_ = original_width;
    }
    if (height_ < 1) {
        height_ = original_height;
    }

    if (ctx_.device_type == kDLGPU) {
        ndarray_pool_ = NDArrayPool(0, {height_, width_, 3}, kUInt8, ctx_);
    }

    decoder_->SetCodecContext(dec_ctx, width_, height_, rotation);
    IndexKeyframes();
}

unsigned int VideoReader::QueryStreams() const {
    if (!fmt_ctx_) return 0;
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; ++i) {
        // iterate and print stream info
        // feel free to add more if needed
        AVStream *st = fmt_ctx_->streams[i];
        const AVCodec *codec = codecs_[i];
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
    if (!fmt_ctx_) return 0;
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
    if (cnt < 1) {
        LOG(FATAL) << "[" << filename_ << "] Failed to measure duration/frame-count due to broken metadata.";
    }
    return cnt;
}

int64_t VideoReader::GetCurrentPosition() const {
    if (!fmt_ctx_) return 0;
    return curr_frame_;
}

int64_t VideoReader::FrameToPTS(int64_t pos) {
    int64_t ts = frame_ts_[pos].pts;
    return ts;
}

std::vector<int64_t> VideoReader::FramesToPTS(const std::vector<int64_t>& positions) {
    std::vector<int64_t> ret;
    ret.reserve(positions.size());
    for (auto pos : positions) {
        ret.emplace_back(frame_ts_[pos].pts);
    }
    return ret;
}

bool VideoReader::Seek(int64_t pos) {
    if (!fmt_ctx_) return false;
    if (curr_frame_ == pos) return true;
    decoder_->Clear();
    cached_frame_ = NDArray();
    eof_ = false;

    int64_t ts = FrameToPTS(pos);
    int flag = curr_frame_ > pos ? AVSEEK_FLAG_BACKWARD : 0;

    int ret = av_seek_frame(fmt_ctx_.get(), actv_stm_idx_, ts, flag);
    if (flag != AVSEEK_FLAG_BACKWARD && ret < 0) {
      ret = av_seek_frame(fmt_ctx_.get(), actv_stm_idx_, ts, AVSEEK_FLAG_BACKWARD);
    }
    if (ret < 0) {
        // final try if all above seek fails
        ret = av_seek_frame(fmt_ctx_.get(), actv_stm_idx_, pos, AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_FRAME);
    }
    if (ret < 0) LOG(WARNING) << "Failed to seek file to position: " << pos;
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
    if (!fmt_ctx_) return false;
    if (curr_frame_ == pos) return true;
    int64_t key_pos = LocateKeyframe(pos);
    int64_t curr_key_pos = LocateKeyframe(curr_frame_);
    overrun_ = false;
    // std::cout << "seek " << pos << "(" << frame_ts_[pos].pts << "), nearest key " << key_pos << "(" << frame_ts_[key_pos].pts << "), current pos "
    // << curr_frame_ << "(" << frame_ts_[curr_frame_].pts << "), current key " << curr_key_pos  << "(" << frame_ts_[curr_key_pos].pts << ")" << std:: endl;
    if (key_pos != curr_key_pos || pos < curr_frame_) {
        // need to seek to keyframes first
        // std::cout << "need to seek to keyframe " << key_pos << " first " << std::endl;
        // first rewind to 0, in order to increase seek accuracy
        bool ret = Seek(0);
        if (!ret) return false;
        ret = Seek(key_pos);
        if (!ret) return false;
        // double check if keyframe was jumpped correctly
        if(CheckKeyFrame()){
            if(pos - key_pos > 0){
                SkipFramesImpl(pos - curr_frame_);
            } else if(pos - key_pos == 0){
                overrun_ = true;
            }
        } else {
            if(curr_frame_ < pos){
                SkipFramesImpl(pos - curr_frame_);
            } else {
                key_pos = LocateKeyframe(pos);
                // since curr_frame_ is larger, Seek will use AVSEEK_FLAG_BACKWARD
                Seek(key_pos);
                SkipFramesImpl(pos - key_pos);
            }
        }
    } else {
        // no need to seek to keyframe, since both current and seek position belong to same keyframe
        SkipFramesImpl(pos - curr_frame_);
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
    if (overrun_)
    {
        overrun_ = false;
        return tmp_key_frame_;
    }
    NDArray frame;
    decoder_->Start();
    bool ret = false;
    int rewind_offset = 0;
    int retry = 0;
    while (!ret) {
        PushNext();
        if (curr_frame_ >= GetFrameCount()) {
            return NDArray::Empty({}, kUInt8, ctx_);
        }
        ret = decoder_->Pop(&frame);
        if (frame.Size() <= 1) {
            if (frame.defined() && frame.data_->dl_tensor.dtype == kInt64) {
              // draining finished
              if (FetchCachedFrame(frame, curr_frame_)) {
                break;
              } else {
                if (rewind_offset > REWIND_RETRY_MAX) {
                  LOG(FATAL) << "[" << filename_ << "]Unable to handle EOF because the video might have corrupted frames" 
                  << "and `DECORD_REWIND_RETRY_MAX=" << REWIND_RETRY_MAX << "`. You may override the limit by `export DECORD_REWIND_RETRY_MAX=32`"
                  << " for example to allow more auto-substituded frames, exit...";
                }
                SeekAccurate(curr_frame_ - rewind_offset);
                ++rewind_offset;
                ret = false;
              }
            } else {
              // skipped frames or waiting for more packets
              if (eof_ && retry > EOF_RETRY_MAX) {
                if (FetchCachedFrame(frame, curr_frame_)) {
                  break;
                } else {
                  LOG(FATAL) << "[" << filename_ << "]Unable to handle EOF because it takes too long to retrieve last few frames and "
                  << "`DECORD_EOF_RETRY_MAX=" << EOF_RETRY_MAX << "`. You may override the limit by `export DECORD_EOF_RETRY_MAX=20480`"
                  << " for example to allow more EOF retry attempts, exit...";
                }
              }
              retry++;
              ret = false;
            }
        }
    }
    if (frame.defined()) {
        ++curr_frame_;
        CacheFrame(frame);
    }
    return frame;
}

NDArray VideoReader::NextFrame() {
    if (!fmt_ctx_) return NDArray();
    return NextFrameImpl();
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
            // std::cout << ((packet->flags & AV_PKT_FLAG_KEY) ? "*" : "") << cnt << ": pts " << packet->pts << ", dts " << packet->dts << ", start pts " << start_pts << ", stop pts " << stop_pts << std::endl;
            if (packet->flags & AV_PKT_FLAG_KEY) {
                key_indices_.emplace_back(cnt);
            }
            ++cnt;
        }
        av_packet_unref(packet.get());
    }
    std::sort(std::begin(frame_ts_), std::end(frame_ts_),
            [](const AVFrameTime& a, const AVFrameTime& b) -> bool
                {return a.pts < b.pts;});

    for (size_t i = 0; i < frame_ts_.size(); ++i){
        pts_frame_map_.insert(std::pair<int64_t, int64_t>(frame_ts_[i].pts, i));
        // std::cout << i << ": pts " << frame_ts_[i].pts << ", dts " << frame_ts_[i].dts << ", start pts " << frame_ts_[i].start << ", stop pts " << frame_ts_[i].stop << std::endl;
    }
    curr_frame_ = GetFrameCount();
    ret = Seek(0);
}

runtime::NDArray VideoReader::GetKeyIndices() {
    if (!fmt_ctx_) return NDArray();
    std::vector<int64_t> shape = {static_cast<int64_t>(key_indices_.size())};
    runtime::NDArray ret = runtime::NDArray::Empty(shape, kInt64, kCPU);
    ret.CopyFrom<int64_t>(key_indices_, shape);
    return ret;
}

runtime::NDArray VideoReader::GetFramePTS() const {
    if (!fmt_ctx_) return NDArray();
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
    if (!fmt_ctx_) return 0.0;
    CHECK(actv_stm_idx_ >= 0);
    CHECK(static_cast<unsigned int>(actv_stm_idx_) < fmt_ctx_->nb_streams);
    AVStream *active_st = fmt_ctx_->streams[actv_stm_idx_];
    return static_cast<double>(active_st->avg_frame_rate.num) / active_st->avg_frame_rate.den;
}

double VideoReader::GetRotation() const {
    if (!fmt_ctx_) return 0.0;
    CHECK(actv_stm_idx_ >= 0);
    CHECK(static_cast<unsigned int>(actv_stm_idx_) < fmt_ctx_->nb_streams);
    AVStream *active_st = fmt_ctx_->streams[actv_stm_idx_];
    AVDictionaryEntry *rotate = av_dict_get(active_st->metadata, "rotate", NULL, 0);

    double theta = 0;
    if (rotate && *rotate->value && strcmp(rotate->value, "0"))
        theta = atof(rotate->value);

    uint8_t* displaymatrix = av_stream_get_side_data(active_st, AV_PKT_DATA_DISPLAYMATRIX, NULL);
    if (displaymatrix && !theta)
        theta = -av_display_rotation_get((int32_t*) displaymatrix);

    theta = std::fmod(theta, 360);
    if(theta < 0) theta += 360;

    return theta;
}

std::vector<int64_t> VideoReader::GetKeyIndicesVector() const {
    return key_indices_;
}

void VideoReader::SkipFrames(int64_t num) {
    if (!fmt_ctx_) return;
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

    SkipFramesImpl(num);
}

bool VideoReader::CheckKeyFrame()
{
    // check curr_frame_ is correct or not, by decoding the current frame
    NDArray frame;
    decoder_->Start();
    bool ret = false;
    int64_t cf = curr_frame_;
    while (!ret)
    {
        PushNext();
        ret = decoder_->Pop(&frame);
    }

    if (eof_ && frame.pts == -1){
        // wrongly jumpped to the end of file
        curr_frame_ = GetFrameCount();
        return false;
    }

    if(frame.pts == -1){
        LOG(FATAL) << "Error seeking keyframe: " << curr_frame_ << " with total frames: " << GetFrameCount();
    }

    // find the real current frame after decoding
    auto iter = pts_frame_map_.find(frame.pts);
    if (iter != pts_frame_map_.end())
        cf = iter->second;
    if (curr_frame_ != cf)
    {
        curr_frame_ = cf + 1;
        return false;
    } else{
        ++curr_frame_;
        tmp_key_frame_ = frame;
        return true;
    }

}

void VideoReader::SkipFramesImpl(int64_t num)
{
    if (!fmt_ctx_)
        return;
    num = std::min(GetFrameCount() - curr_frame_, num);
    if (num < 1) return;

    NDArray frame;
    decoder_->Start();
    bool ret = false;
    std::vector<int64_t> frame_pos(num);
    std::iota(frame_pos.begin(), frame_pos.end(), curr_frame_);
    auto pts = FramesToPTS(frame_pos);
    decoder_->SuggestDiscardPTS(pts);


    int64_t pop_retries = 0;
    const int64_t MAX_POP_RETRIES_PER_FRAME = 10000;
    int64_t initial_num = num;

    while (num > 0) {
        PushNext();
        ret = decoder_->Pop(&frame);
        if (!ret) {
            pop_retries++;
            if (pop_retries > MAX_POP_RETRIES_PER_FRAME) {
                LOG(INFO) << "[" << filename_ << "] Failed to skip frames effectively at frame " << curr_frame_
                           << ". Decoder did not respond after " << MAX_POP_RETRIES_PER_FRAME
                           << " attempts. Video might be corrupted or seeking failed. Aborting skip operation."
                           << " Attempted to skip " << initial_num << " frames, skipped " << (initial_num - num) << ".";
                break;
            }
            continue;
        }
        ++curr_frame_;
        // LOG(INFO) << "skip: " << num;
        --num;
    }
    decoder_->ClearDiscardPTS();
}

NDArray VideoReader::GetBatch(std::vector<int64_t> indices, NDArray buf) {
    if (!fmt_ctx_) return NDArray();
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
            if (curr_frame_ != pos) {
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

void VideoReader::CacheFrame(NDArray frame) {
    if (!use_cached_frame_) return;
    if (!cached_frame_.defined()) {
        cached_frame_ = NDArray::Empty({height_, width_, 3}, kUInt8, ctx_);
    }
    if (!frame.defined()) return;
    if (cached_frame_.Size() != frame.Size()) return;
    cached_frame_.CopyFrom(frame);
}

bool VideoReader::FetchCachedFrame(NDArray &frame, int64_t pos) {
  if (!use_cached_frame_) return false;
  if (cached_frame_.Size() <= 1) return false;
  if (!frame.defined() || frame.Size() != cached_frame_.Size()) {
      frame = NDArray::Empty({height_, width_, 3}, kUInt8, ctx_);
  }
  frame.CopyFrom(cached_frame_);
  failed_idx_.insert(pos);
  int64_t failed_count = failed_idx_.size();
  if (fault_tol_thresh_ >= 0) {
      if (failed_count > fault_tol_thresh_) {
          LOG(FATAL) << "[" << filename_ << "]You have received more than " << fault_tol_thresh_
            << " duplicate frames that are corrupted and recovered from nearest frames.";
      }
  }
  if (failed_count > DUPLICATE_WARNING_THRESHOLD * GetFrameCount()) {
      if (!fault_warn_emit_) {
          LOG(WARNING) << "[" << filename_ << "]You have received more than " << failed_idx_.size()
            << " frames corrupted and recovered from nearest frames."
            << " Set environment variable `DECORD_DUPLICATE_WARNING_THRESHOLD=1.0`"
            << "if you want to disable this warning.";
          fault_warn_emit_ = true;
      }
  }
  return true;
}

}  // namespace decord
