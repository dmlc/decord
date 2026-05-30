/*!
 *  Copyright (c) 2024 by Contributors if not otherwise specified
 * \file videotoolbox_threaded_decoder.h
 * \brief VideoToolbox based decoder for macOS GPU acceleration
 */

#ifndef DECORD_VIDEO_VIDEOTOOLBOX_VIDEOTOOLBOX_THREADED_DECODER_H_
#define DECORD_VIDEO_VIDEOTOOLBOX_VIDEOTOOLBOX_THREADED_DECODER_H_

#include "../ffmpeg/ffmpeg_common.h"
#include "../threaded_decoder_interface.h"

#include <condition_variable>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <decord/runtime/ndarray.h>
#include <dmlc/concurrency.h>
#include <dlpack/dlpack.h>

#ifdef __APPLE__
#include <VideoToolbox/VideoToolbox.h>
#include <CoreVideo/CoreVideo.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#endif

namespace decord {
namespace videotoolbox {

class VideoToolboxThreadedDecoder final : public ThreadedDecoderInterface {
    constexpr static int kMaxOutputSurfaces = 20;
    using NDArray = runtime::NDArray;
    using AVPacketPtr = ffmpeg::AVPacketPtr;
    using AVCodecContextPtr = ffmpeg::AVCodecContextPtr;
    using AVBSFContextPtr = ffmpeg::AVBSFContextPtr;
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;

    public:
        VideoToolboxThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat);
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1, int rotation = 0);
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt, NDArray buf);
        bool Pop(NDArray *frame);
        void SuggestDiscardPTS(std::vector<int64_t> dts);
        void ClearDiscardPTS();
        ~VideoToolboxThreadedDecoder();

        // VideoToolbox callback functions
        static void VTDecompressionOutputCallback(void *decompressionOutputRefCon,
                                                  void *sourceFrameRefCon,
                                                  OSStatus status,
                                                  VTDecodeInfoFlags infoFlags,
                                                  CVImageBufferRef imageBuffer,
                                                  CMTime presentationTimeStamp,
                                                  CMTime presentationDuration);

    private:
        void LaunchThread();
        void LaunchThreadImpl();
        void RecordInternalError(std::string message);
        void CheckErrorStatus();
        void InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat);
        NDArray ConvertCVImageBufferToNDArray(CVImageBufferRef imageBuffer);
        bool SetupVideoToolboxDecoder(AVCodecParameters *codecpar);
        void CleanupVideoToolboxDecoder();
#ifdef __APPLE__
        void DecodePacket(AVPacket *pkt);
        CMVideoCodecType DetectProResVariant(AVCodecParameters *codecpar);
#endif

        int device_id_;
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        std::thread launcher_t_;
        std::atomic<bool> run_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;
        std::atomic<bool> initialized_;

        AVCodecContextPtr dec_ctx_;
        AVBSFContextPtr bsf_ctx_;
        unsigned int width_;
        unsigned int height_;

        // VideoToolbox specific
#ifdef __APPLE__
        VTDecompressionSessionRef decompression_session_;
        CMFormatDescriptionRef format_description_;
        std::mutex vt_session_mutex_;
#endif

        std::unordered_set<int64_t> discard_pts_;
        std::mutex pts_mutex_;
        std::mutex error_mutex_;
        std::atomic<bool> error_status_;
        std::string error_message_;

        // Frame ordering and timing
        AVRational vt_time_base_;
        AVRational frame_base_;
        std::unordered_map<int64_t, NDArray> frame_buffer_;
        std::mutex frame_buffer_mutex_;

    DISALLOW_COPY_AND_ASSIGN(VideoToolboxThreadedDecoder);
};

}  // namespace videotoolbox
}  // namespace decord

#endif  // DECORD_VIDEO_VIDEOTOOLBOX_VIDEOTOOLBOX_THREADED_DECODER_H_