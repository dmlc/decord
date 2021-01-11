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
        NDArrays GetBatch(std::vector<int> indices, NDArrays buffer);
    private:
        int Decode(std::string fn);
        int DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex);
        int ToNDArray();
        int Resample(int sampleRate, int numChannels);

        AVFormatContext *pFormatContext;
        std::vector<AVCodec*> codecs;
        std::vector<AVCodecParameters*> codecParameters;
        std::vector<int> audioStreamIndices;
        std::vector<std::unique_ptr<AudioStream>> audios;
        std::vector<NDArray> audioOutputs;
        std::string filename;
        int sampleRate;
        int numChannels;
    };

}

#endif //DECORD_AUDIO_INTERFACE_H
