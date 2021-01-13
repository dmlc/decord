//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AUDIO_READER_H_
#define DECORD_AUDIO_READER_H_

#include <libswresample/swresample.h>
#include <vector>

#include "../../include/decord/audio_interface.h"
#include "audio_stream.h"

namespace decord {

    class AudioReader: public AudioReaderInterface {
    public:
        AudioReader(std::string fn, int sampleRate, int numChannels);
        ~AudioReader();
        NDArray GetBatch(std::vector<int> indices, NDArray buffer);
    private:
        int Decode(std::string fn);
        int DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex);
        void handleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame);
        void drainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame);
        int ToNDArray();
        int Resample(int sampleRate, int numChannels);

        AVFormatContext *pFormatContext;
        std::vector<AVCodec*> codecs;
        std::vector<AVCodecParameters*> codecParameters;
        std::vector<int> audioStreamIndices;
        std::vector<std::unique_ptr<AudioStream>> audios;
        NDArray audioOutputs;
        std::string filename;
        int sampleRate;
        int numChannels;
    };

}

#endif //DECORD_AUDIO_INTERFACE_H
