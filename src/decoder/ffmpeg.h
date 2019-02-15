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
struct AVPacket_ {
    AVPacket_() { ptr = std::shared_ptr<AVPacket>(av_packet_alloc(), &Deleter);}
    AVPacket_(AVPacket *p) { ptr = std::shared_ptr<AVPacket>(p, &Deleter); }
    AVPacket_(const AVPacket_& other) { ptr = other.ptr; }
    AVPacket_(AVPacket_&& other) { std::swap(ptr, other.ptr); }
    void *Deleter(AVPacket_ *self) { AVPacket *p = self->ptr.get(); av_packet_free(&p); ptr = nullptr; };
    std::shared_ptr<AVPacket> ptr;
}; // struct AVPacket_

// AVFrame wrapper with smart pointer 
struct AVFrame_ {
    AVFrame_() { ptr = std::shared_ptr<AVFrame>(av_frame_alloc(), &Deleter); }
    AVFrame_(AVFrame *p) { ptr = std::shared_ptr<AVFrame>(p, &Deleter); }
    AVFrame_(const AVFrame_& other) { ptr = other.ptr; }
    AVFrame_(AVFrame_&& other) { std::swap(ptr, other.ptr); }
    void *Deleter(AVFrame_ *self) { AVFrame *p = self->ptr.get(); av_frame_free(&p); ptr = nullptr; };
    std::shared_ptr<AVFrame> ptr;
}; // struct AVFrame_

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

class FFMPEGVideoReader : public VideoReaderInterface {
    public:

    private:
        FFMPEGVideoDecoder dec_;
};

class FFMPEGPacketDispatcher {
    public:

    private:
        dmlc::ConcurrentBlockingQueue<AVPacket_> pkt_queue_;
};

class FFMPEGThreadedDecoder {
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacket_>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<AVFrame_>;
    public:
        FFMPEGThreadedDecoder();
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacket *pkt);
        void Push(AVPacket_ pkt);
        bool Pull(AVFrame *frame);
        bool Pull(AVFrame_ frame);
        ~FFMPEGThreadedDecoder();
    protected:
        AVCodecContext *dec_ctx_;
    private:
        void DecodePacket(PacketQueue& pkt_queue, FrameQueue& frame_queue, std::atomic_bool& run);
        PacketQueue pkt_queue_;
        FrameQueue frame_queue_;
        std::thread t_;
        std::atomic_bool run_;
};

class FFMPEGVideoDecoder {
    public:
        FFMPEGVideoDecoder(std::string& fn);
        /*! \brief Destructor, note that FFMPEG resources has to be managed manually to avoid resource leak */
        ~FFMPEGVideoDecoder();
        void SetVideoStream(int stream_nb = -1);
        unsigned int QueryStreams() const;
        runtime::NDArray NextFrame();
    protected:
        friend class FFMPEGVideoReader;
        void Reset();
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