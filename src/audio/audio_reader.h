//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AUDIO_READER_H_
#define DECORD_AUDIO_READER_H_

#include <vector>

#include "../../include/decord/audio_interface.h"

namespace decord {

    class AudioReader: public AudioReaderInterface {
    public:
        AudioReader(std::string fn, int sampleRate, DLContext ctx, int io_type=kNormal, bool mono=true);
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
        std::unique_ptr<ffmpeg::AVIOBytesContext> io_ctx_;  // avio context for raw memory access
        AVFormatContext *pFormatContext;
        struct SwrContext* swr;
        // AVCodec* pCodec;
        AVCodecParameters* pCodecParameters;
        AVCodecContext * pCodecContext;
        int audioStreamIndex;
//        std::vector<std::unique_ptr<AudioStream>> audios;
        std::vector<std::vector<float>> outputVector;
        NDArray output;
        // padding is the start time in seconds of the first audio sample
        double padding;
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

}

#endif //DECORD_AUDIO_INTERFACE_H
