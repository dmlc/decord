//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AUDIO_READER_H_
#define DECORD_AUDIO_READER_H_

#include <vector>
extern "C"
{
#include <libavutil/channel_layout.h>  // Necesario para AVChannelLayout
}

#include "../../include/decord/audio_interface.h"

namespace decord {

    class AudioReader : public AudioReaderInterface {
    public:
        AudioReader(std::string fn, int sampleRate, DLContext ctx, int io_type = kNormal, bool mono = true);
        ~AudioReader();
        NDArray GetNDArray();
        int GetNumPaddingSamples();
        double GetDuration();
        int64_t GetNumSamplesPerChannel();
        int GetNumChannels();
        void GetInfo();

    private:
        int Decode(std::string fn, int io_type);
        void DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex);
        void HandleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame);
        void DrainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame);
        void InitSWR(AVCodecContext *pCodecContext);
        void ToNDArray();
        void SaveToVector(float** buffer, int numChannels, int numSamples);

        DLContext ctx;
        std::unique_ptr<ffmpeg::AVIOBytesContext> io_ctx_;  // AVIO context para acceso a memoria raw
        AVFormatContext *pFormatContext;
        struct SwrContext* swr;
        AVCodecParameters* pCodecParameters;
        AVCodecContext *pCodecContext;
        int audioStreamIndex;
        std::vector<std::vector<float>> outputVector;
        NDArray output;
        double padding;  // Tiempo de inicio en segundos de la primera muestra de audio
        std::string filename;
        int originalSampleRate;
        int targetSampleRate;
        int numChannels;
        bool mono;
        int totalSamplesPerChannel;
        int totalConvertedSamplesPerChannel;
        double timeBase;
        double duration;
    };

}  // namespace decord

#endif  // DECORD_AUDIO_READER_H_