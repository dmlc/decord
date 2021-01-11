//
// Created by Yin, Weisu on 1/8/21.
//

#include "audio_reader.h"

#include <memory>

namespace decord {

    AudioReader::AudioReader(std::string fn, int sampleRate, int numChannels)
    : pFormatContext(nullptr), codecs(), filename(fn), sampleRate(sampleRate), numChannels(numChannels) {
        Decode(fn);
        // if sampleRate does not equal to stream sampleRate of each strea
        {
            Resample(sampleRate, numChannels);
        }
        ToNDArray();
    }

    int AudioReader::Decode(std::string fn) {
        // Get format context
        pFormatContext = avformat_alloc_context();
        // Open input
        avformat_open_input(&pFormatContext, fn.c_str(), NULL, NULL);
        // Read stream
        avformat_find_stream_info(pFormatContext, NULL);

        for (auto i = 0; i < pFormatContext->nb_streams; i++) {
            // find the stream needed
            AVCodecParameters *tempCodecParameters = pFormatContext->streams[i]->codecpar;
            if (tempCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndices.push_back(i);
                int sr = tempCodecParameters->sample_rate;
                int nc = tempCodecParameters->channels;
                int duration = pFormatContext->streams[i]->duration;
                std::unique_ptr<AudioStream> stream(new AudioStream(sr, nc, duration));
                audios.push_back(std::move(stream));
                codecParameters.push_back(tempCodecParameters);
            }
        }
        // For each stream, open codec and decode
        for (auto i = 0; i < audioStreamIndices.size(); i++) {
            // prepare codec
            AVCodec *pCodec = nullptr;
            AVCodecParameters *pCodecParameters = codecParameters[i];
            pCodec = avcodec_find_decoder(pCodecParameters->codec_id);
            AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
            avcodec_parameters_to_context(pCodecContext, pCodecParameters);
            avcodec_open2(pCodecContext, pCodec, NULL);
            // prepare packet and frame
            AVPacket *pPacket = av_packet_alloc();
            AVFrame *pFrame = av_frame_alloc();
            DecodePacket(pPacket, pCodecContext, pFrame, audioStreamIndices[i]);
        }

        return 0;
    }

    int AudioReader::DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex) {
        // Get the packet
        // Check if the packet belongs to the stream we want
        // Send packet to the decoder
        // Receive the decoded frames
        return 0;
    }

    int AudioReader::Resample(int sampleRate, int numChannels) {

        return 0;
    }

    int AudioReader::ToNDArray() {

        return 0;
    }
}