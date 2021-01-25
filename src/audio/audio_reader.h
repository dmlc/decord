//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AUDIO_READER_H_
#define DECORD_AUDIO_READER_H_

#include <vector>

#include "../../include/decord/audio_interface.h"
#include "audio_stream.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <libswresample/swresample.h>
#ifdef __cplusplus
}
#endif
#include <fstream>

namespace decord {

    class AudioReader: public AudioReaderInterface {
    public:
        AudioReader(std::string fn, int sampleRate, DLContext ctx);
        ~AudioReader();
        NDArray GetNDArray();
        int GetNumPaddingSamples();
        double GetDuration();
        int64_t GetNumSamplesPerChannel();
        int GetNumChannels();
    private:
        void Decode(std::string fn);
        void DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex);
        void HandleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame);
        void DrainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame);
        void InitSWR(AVCodecContext *pCodecContext);
        void ToNDArray();
        void SaveToVector(float** buffer, int numChannels, int numSamples);

        DLContext ctx;
        AVFormatContext *pFormatContext;
        struct SwrContext* swr;
        AVCodec* pCodec;
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
        int totalSamplesPerChannel;
        int totalConvertedSamplesPerChannel;
        double timeBase;
        double duration;
        std::ofstream outfile;
    };

}

#endif //DECORD_AUDIO_INTERFACE_H
