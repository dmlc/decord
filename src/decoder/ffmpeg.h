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
#include <thread>
// #include <condition_variable>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#ifdef __cplusplus
}
#endif
#include <decord/base.h>
#include <dmlc/concurrency.h>


namespace decord {
namespace ffmpeg {
// namespace wrapper {
// class AVPacket_ {
//     public:
//         AVPacket_();
//         AVPacket_(const AVPacket_ &pkt);
//         AVPacket_(AVPacket_ &&pkt);
//         explicit AVPacket_(const AVPacket *pkt);
//         ~AVPacket_();
//     private:
        
// };
// }  // namespace wrapper, for FFMPEG C++ wrapper

// AVPacket wrapper with smart pointer 
// struct AVPacket_ {
//     AVPacket_() : ptr(nullptr) {}
//     AVPacket_(AVPacket *p) { ptr = std::shared_ptr<AVPacket>(p, &AVPacket_::Deleter); }
//     AVPacket_(const AVPacket_& other) { ptr = other.ptr; }
//     AVPacket_(AVPacket_&& other) { std::swap(ptr, other.ptr); }
//     AVPacket_& operator= ( const AVPacket_ &other ) = default;
//     AVPacket* Get() { return ptr.get(); }
//     void Alloc() { ptr = std::shared_ptr<AVPacket>(av_packet_alloc(), &AVPacket_::Deleter); }
//     static void Deleter(AVPacket *p) { if (!p) return; av_packet_free(&p); p = nullptr; }
//     std::shared_ptr<AVPacket> ptr;
// }; // struct AVPacket_

// // AVFrame wrapper with smart pointer 
// struct AVFrame_ {
//     AVFrame_() : ptr(nullptr) {}
//     AVFrame_(AVFrame *p) { ptr = std::shared_ptr<AVFrame>(p, &AVFrame_::Deleter); }
//     AVFrame_(const AVFrame_& other) { ptr = other.ptr; }
//     AVFrame_(AVFrame_&& other) { std::swap(ptr, other.ptr); }
//     AVFrame_& operator= ( const AVFrame_ &other ) = default;
//     AVFrame* Get() { return ptr.get(); }
//     void Alloc() { ptr = std::shared_ptr<AVFrame>(av_frame_alloc(), &AVFrame_::Deleter); }
//     static void Deleter(AVFrame *p) { if (!p) return; av_frame_free(&p); p = nullptr; }
//     DLTensor ToDLTensor();
//     std::shared_ptr<AVFrame> ptr;
//     int64_t shape[3];
// }; // struct AVFrame_

// // AVCodecContext wrapper with smart pointer
// struct AVCodecContext_ {
//     AVCodecContext_() : ptr(nullptr) {}
//     AVCodecContext_(AVCodecContext *p) : ptr(p, &AVCodecContext_::Deleter) { }
//     AVCodecContext* Get() { return ptr.get(); }
//     static void Deleter(AVCodecContext *p) { if (!p) return; avcodec_free_context(&p); p = nullptr; }
//     std::shared_ptr<AVCodecContext> ptr;
// };  // AVCodecContext_

// SwsContext wrapper
// struct SwsContext_ {
//     SwsContext_() : ptr(nullptr) {}
//     SwsContext_(SwsContext *p) { ptr = std::make_shared<SwsContext>(p, &SwsContext_::Deleter); }
//     static void Deleter(SwsContext_ *self) { if (!self->ptr) return; sws_freeContext(self->ptr.get()); self->ptr = nullptr; }
//     std::shared_ptr<SwsContext> ptr;
// };  // SwsContext_

class FFMPEGVideoDecoder;
class FFMPEGVideoReader;

/*! \brief FrameTransform as map key */
struct FrameTransform {
    AVPixelFormat fmt;
    uint32_t height;
    uint32_t width;
    uint32_t channel;
    int interp;  // interpolation method
    explicit FrameTransform(DLDataType dtype, uint32_t h, uint32_t w, uint32_t c, int interp);
};  // struct FrameTransform


class FFMPEGFilterGraph {
    public:
        FFMPEGFilterGraph(std::string filter_desc, AVCodecContext *dec_ctx);
        void Push(AVFrame *frame);
        bool Pop(AVFrame **frame);
        ~FFMPEGFilterGraph();
    private:
        void Init(std::string filter_desc, AVCodecContext *dec_ctx);
        AVFilterContext *buffersink_ctx_;
        AVFilterContext *buffersrc_ctx_;
        AVFilterGraph *filter_graph_;
        std::atomic<int> count_;

    DISALLOW_COPY_AND_ASSIGN(FFMPEGFilterGraph);
};  // FFMPEGFilterGraph

class FFMPEGThreadedDecoder {
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacket*>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<AVFrame*>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using FFMPEGFilterGraphPtr = std::shared_ptr<FFMPEGFilterGraph>;
    public:
        FFMPEGThreadedDecoder();
        void SetCodecContext(AVCodecContext *dec_ctx);
        void Start();
        void Stop();
        void Clear();
        // void Push(AVPacket *pkt);
        void Push(AVPacket* pkt);
        // bool Pull(AVFrame *frame);
        bool Pop(AVFrame **frame);
        ~FFMPEGThreadedDecoder();
    protected:
        friend class FFMPEGVideoReader;
        
        // SwsContext_ sws_ctx_;
    private:
        void WorkerThread();
        // void FetcherThread(std::condition_variable& cv, FrameQueuePtr frame_queue);
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        std::thread t_;
        // std::thread fetcher_;
        // std::condition_variable cv_;
        std::atomic<bool> run_;
        FFMPEGFilterGraphPtr filter_graph_;
        AVCodecContext *dec_ctx_;

    DISALLOW_COPY_AND_ASSIGN(FFMPEGThreadedDecoder);
};

class FFMPEGVideoReader : public VideoReaderInterface {
    using FFMPEGThreadedDecoderPtr = std::unique_ptr<FFMPEGThreadedDecoder>;
    public:
        FFMPEGVideoReader(std::string fn);
        /*! \brief Destructor, note that FFMPEG resources has to be managed manually to avoid resource leak */
        ~FFMPEGVideoReader();
        void SetVideoStream(int stream_nb = -1);
        unsigned int QueryStreams() const;
        runtime::NDArray NextFrame();
    protected:
        // void Reset();
    private:
        /*! \brief Get or Create SwsContext by dtype */
        // struct SwsContext* GetSwsContext(FrameTransform out_fmt);
        /*! \brief Video Streams Codecs in original videos */
        std::vector<AVCodec*> codecs_;
        /*! \brief Currently active video stream index */
        int actv_stm_idx_;
        /*! \brief AV format context holder */
        AVFormatContext *fmt_ctx_;
        /*! \brief AVPacket buffer */
        // AVPacket *pkt_;
        /*! \brief AVFrame buffer */
        // AVFrame *frame_;
        /*! \brief AV dodec context for decoding related info */
        // AVCodecContext *dec_ctx_;
        /*! \brief Container for various FFMPEG swsContext */
        // std::unordered_map<FrameTransform, struct SwsContext*> sws_ctx_map_;
        FFMPEGThreadedDecoderPtr decoder_;

};  // class FFMPEGVideoReader


}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_BACKEND_FFMPEG_H_