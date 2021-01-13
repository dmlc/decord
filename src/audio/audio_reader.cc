//
// Created by Yin, Weisu on 1/8/21.
//

#include "audio_reader.h"
#include <memory>
extern "C" {
#include <libavutil/error.h>
#include <libavutil/frame.h>
}

namespace decord {

    AudioReader::AudioReader(std::string fn, int sampleRate, int numChannels)
    : pFormatContext(nullptr), codecs(), filename(fn), sampleRate(sampleRate), numChannels(numChannels) {
        if (Decode(fn) != 0) {
            LOG(FATAL) << "ERROR failed to decode audio";
        };
        ToNDArray();
    }

    int AudioReader::Decode(std::string fn) {
        // Get format context
        pFormatContext = avformat_alloc_context();
        CHECK(pFormatContext != nullptr) << "Unable to alloc avformat context";
        // Open input
        // TODO: Support Raw Bytes
        int formatOpenRet = 1;
        formatOpenRet = avformat_open_input(&pFormatContext, fn.c_str(), NULL, NULL);
        if (formatOpenRet != 0) {
            char errstr[200];
            av_strerror(formatOpenRet, errstr, 200);
            LOG(WARNING) << "ERROR opening" << fn.size() << " bytes, " << errstr;
            return -1;
        }
        // Read stream
        avformat_find_stream_info(pFormatContext, NULL);

        bool streamFound = false;
        for (auto i = 0; i < pFormatContext->nb_streams; i++) {
            // find the stream needed
            AVCodecParameters *tempCodecParameters = pFormatContext->streams[i]->codecpar;
            if (tempCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
                streamFound = true;
                audioStreamIndices.push_back(i);
                int sr = tempCodecParameters->sample_rate;
                int nc = tempCodecParameters->channels;
                int duration = pFormatContext->streams[i]->duration;
                std::unique_ptr<AudioStream> stream(new AudioStream(sr, nc, duration));
                audios.push_back(std::move(stream));
                codecParameters.push_back(tempCodecParameters);
            }
        }
        if (!streamFound) { LOG(WARNING) << "Can't find audio stream"; }
        // For each stream, open codec and decode
        for (auto i = 0; i < audioStreamIndices.size(); i++) {
            // prepare codec
            AVCodec *pCodec = nullptr;
            AVCodecParameters *pCodecParameters = codecParameters[i];
            pCodec = avcodec_find_decoder(pCodecParameters->codec_id);
            CHECK(pCodec != nullptr) << "ERROR Decoder not found. THe codec is not supported.";
            AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
            CHECK(pCodecContext != nullptr) << "ERROR Could not allocate a decoding context.";
            CHECK_GE(avcodec_parameters_to_context(pCodecContext, pCodecParameters), 0) << "ERROR Could not set context parameters.";
            int codecOpenRet = 1;
            codecOpenRet = avcodec_open2(pCodecContext, pCodec, NULL);
            if (codecOpenRet < 0) {
                char errstr[200];
                av_strerror(codecOpenRet, errstr, 200);
                LOG(FATAL) << "ERROR open codec through avcodec_open2: " << errstr;
                avcodec_close(pCodecContext);
                avcodec_free_context(&pCodecContext);
                avformat_close_input(&pFormatContext);
                return -1;
            }
            // prepare packet and frame
            AVPacket *pPacket = av_packet_alloc();
            AVFrame *pFrame = av_frame_alloc();
            DecodePacket(pPacket, pCodecContext, pFrame, audioStreamIndices[i]);
        }

        return 0;
    }

    int AudioReader::DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex) {
        int ret = 0;
        // Get the packet
        int pktRet = -1;
        while ((pktRet = av_read_frame(pFormatContext, pPacket)) != AVERROR_EOF) {
            if (pktRet != 0) {
                LOG(WARNING) << "ERROR Fail to get packet.";
                ret = -1;
                break;
            }
            // Check if the packet belongs to the stream we want
            if (pPacket->stream_index != streamIndex) {
                av_packet_unref(pPacket);
                continue;
            }
            // Send packet to the decoder
            int sendRet = -1;
            sendRet = avcodec_send_packet(pCodecContext, pPacket);
            if (sendRet != 0) {
                // EAGAIN shouldn't be treat as an error
                if (sendRet != AVERROR(EAGAIN)) {
                    LOG(WARNING) << "ERROR Fail to send packet.";
                    av_packet_unref(pPacket);
                    ret = -1;
                    break;
                }
            }
            // Packet sent successfully, dont need it anymore
            av_packet_unref(pPacket);
            // Receive the decoded frames
            int receiveRet = -1;
            receiveRet = avcodec_receive_frame(pCodecContext, pFrame);
            while (receiveRet == 0) {
                // Handle received frames
                handleFrame(pCodecContext, pFrame);
            }
            if (receiveRet != AVERROR(EAGAIN)) {
                LOG(WARNING) << "ERROR Fail to receive frame.";
                ret = -1;
                break;
            }
        }
        // Drain the decoder
        drainDecoder(pCodecContext, pFrame);

        return ret;
    }

    void AudioReader::handleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame) {

    }

    void AudioReader::drainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame) {

    }

    int AudioReader::Resample(int sampleRate, int numChannels) {

        return 0;
    }

    int AudioReader::ToNDArray() {

        return 0;
    }

    NDArray AudioReader::GetBatch(std::vector<int> indices, NDArray buffer) {
        return NDArray();
    }

    AudioReader::~AudioReader() {

    }
}