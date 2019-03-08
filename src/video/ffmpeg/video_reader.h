/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief FFmpeg video reader, implements VideoReaderInterface
 */

#ifndef DECORD_VIDEO_FFMPEG_VIDEO_READER_H_
#define DECORD_VIDEO_FFMPEG_VIDEO_READER_H_

#include "threaded_decoder.h"
#include <decord/video_interface.h>

#include <string>
#include <vector>
// #include <condition_variable>

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


// /*! \brief FrameTransform as map key */
// struct FrameTransform {
//     AVPixelFormat fmt;
//     uint32_t height;
//     uint32_t width;
//     uint32_t channel;
//     int interp;  // interpolation method
//     explicit FrameTransform(DLDataType dtype, uint32_t h, uint32_t w, uint32_t c, int interp);
// };  // struct FrameTransform




class FFMPEGVideoReader : public VideoReaderInterface {
    using FFMPEGThreadedDecoderPtr = std::unique_ptr<FFMPEGThreadedDecoder>;
    public:
        FFMPEGVideoReader(std::string fn, int width=-1, int height=-1);
        /*! \brief Destructor, note that FFMPEG resources has to be managed manually to avoid resource leak */
        ~FFMPEGVideoReader();
        void SetVideoStream(int stream_nb = -1);
        unsigned int QueryStreams() const;
        runtime::NDArray NextFrame();
    protected:
        // void Reset();
    private:
        void PushNext();
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
        int width_;   // output video width
        int height_;  // output video height
};  // class FFMPEGVideoReader


}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_VIDEO_FFMPEG_VIDEO_READER_H_