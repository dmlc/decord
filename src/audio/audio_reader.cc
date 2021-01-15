//
// Created by Yin, Weisu on 1/8/21.
//

#include "audio_reader.h"
#include <memory>

namespace decord {

    AudioReader::AudioReader(std::string fn, int sampleRate, int numChannels)
    : pFormatContext(nullptr), swr(nullptr), codecs(), filename(fn), sampleRate(sampleRate), numChannels(numChannels) {
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
                LOG(WARNING) << "ERROR open codec through avcodec_open2: " << errstr;
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
        // initialize resample context
        initSWR(pCodecContext);

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
            while ((receiveRet = avcodec_receive_frame(pCodecContext, pFrame)) == 0) {
                // Handle received frames
                handleFrame(pCodecContext, pFrame);
                av_frame_unref(pFrame);
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
        int ret = 0;
        // allocate resample buffer
        uint8_t** outBuffer;
        int outLinesize = 0;
        int outNumChannel = av_get_channel_layout_nb_channels(pFrame->channel_layout);
        int outNumSamples = av_rescale_rnd(swr_get_delay(swr, pFrame->sample_rate) + pFrame->nb_samples,
                                           this->sampleRate, pFrame->sample_rate, AV_ROUND_UP);
        if ((ret = av_samples_alloc_array_and_samples(&outBuffer, &outLinesize, outNumChannel, outNumSamples,
                                                      AV_SAMPLE_FMT_U8P, 0)) < 0)
        {
            LOG(FATAL) << "ERROR Failed to allocate resample buffer";
        }
        int gotSamples = 0;
        gotSamples = swr_convert(swr, outBuffer, outNumSamples, (const uint8_t**)pFrame->extended_data, pFrame->nb_samples);
        CHECK_GE(gotSamples, 0) << "ERROR Failed to convert samples";
        while (gotSamples > 0) {
            // flush buffer
            gotSamples = swr_convert(swr, outBuffer, outNumSamples, NULL, 0);
        }
        // Convert to NDArray
//        switch(pCodecContext->sample_fmt) {
//            case AV_SAMPLE_FMT_U8:
//                // do nothing
//                break;
//            case AV_SAMPLE_FMT_U8P:
//                break;
//            case AV_SAMPLE_FMT_S16:
//                int16_t **data = (int16_t**)pFrame->extended_data;
//                break;
//            case AV_SAMPLE_FMT_S16P:
//                break;
//            case AV_SAMPLE_FMT_S32:
//                break;
//            case AV_SAMPLE_FMT_S32P:
//                break;
//            case AV_SAMPLE_FMT_S64:
//                break;
//            case AV_SAMPLE_FMT_S64P:
//                break;
//            case AV_SAMPLE_FMT_FLT:
//                break;
//            case AV_SAMPLE_FMT_FLTP:
//                break;
//            case AV_SAMPLE_FMT_DBL:
//                break;
//            case AV_SAMPLE_FMT_DBLP:
//                break;
//        }
        // Nonplanar means data for all channels are interleaved together
//        if (!av_sample_fmt_is_planar(pCodecContext->sample_fmt)) {
//            int frameChannels = pFrame->channels;
//            int frameSamples = pFrame->nb_samples;
//            switch (pCodecContext->sample_fmt) {
//                case AV_SAMPLE_FMT_U8P: {
//                    uint8_t buffer[frameChannels][frameSamples];
//                    for (int c = 0; c < frameChannels; c++) {
//                        for (int s = 0; s < frameSamples; s++) {
//                            buffer[c][s] = ((uint8_t**)pFrame->extended_data)[0][s * frameChannels + c];
//                        }
//                    }
//                    // convert buffer to ndarray
//                    break;
//                }
//            }
//        }
    }

    void AudioReader::drainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame) {
        int ret = 0;
        ret = avcodec_send_packet(pCodecContext, NULL);
        if (ret != 0) {
            LOG(WARNING) << "Failed to send packet while draining";
            return;
        }
        int receiveRet = -1;
        while ((receiveRet = avcodec_receive_frame(pCodecContext, pFrame)) == 0) {
            // Handle received frames
            handleFrame(pCodecContext, pFrame);
            av_frame_unref(pFrame);
        }
        if (receiveRet != AVERROR(EAGAIN) && receiveRet != AVERROR_EOF) {
            LOG(WARNING) << "ERROR Fail to receive frame.";
            ret = -1;
        }
    }

    void AudioReader::initSWR(AVCodecContext *pCodecContext) {
        int ret = 0;
        // Set resample ctx
        struct SwrContext* swr = swr_alloc();
        if (!swr) {
            LOG(FATAL) << "ERROR Failed to allocate resample context";
        }
        av_opt_set_channel_layout(swr, "in_channel_layout",  pCodecContext->channel_layout, 0);
        av_opt_set_channel_layout(swr, "out_channel_layout", pCodecContext->channel_layout,  0);
        av_opt_set_int(swr, "in_sample_rate",     pCodecContext->sample_rate,                0);
        av_opt_set_int(swr, "out_sample_rate",    this->sampleRate,                0);
        av_opt_set_sample_fmt(swr, "in_sample_fmt",  pCodecContext->sample_fmt, 0);
        av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_U8P,  0);
        if ((ret = swr_init(swr)) < 0) {
            LOG(FATAL) << "ERROR Failed to initialize resample context";
        }
    }

    int AudioReader::Resample(int sampleRate, int numChannels) {
        struct SwrContext* swr = swr_alloc();
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