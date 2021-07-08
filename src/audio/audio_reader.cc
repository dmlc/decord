//
// Created by Yin, Weisu on 1/8/21.
//

#include "audio_reader.h"
#include "../runtime/str_util.h"
#include <memory>
#include <cmath>

namespace decord {
    // AVIO buffer size when reading from raw bytes
    static const int AVIO_BUFFER_SIZE = std::stoi(runtime::GetEnvironmentVariableOrDefault("DECORD_AVIO_BUFFER_SIZE", "40960"));

    AudioReader::AudioReader(std::string fn, int sampleRate, DLContext ctx, int io_type, bool mono)
    : ctx(ctx), io_ctx_(), pFormatContext(nullptr), swr(nullptr), pCodecParameters(nullptr),
      pCodecContext(nullptr), audioStreamIndex(-1), outputVector(), output(), padding(-1.0), filename(fn), originalSampleRate(0),
      targetSampleRate(sampleRate), numChannels(0), mono(mono), totalSamplesPerChannel(0), totalConvertedSamplesPerChannel(0),
      timeBase(0.0), duration(0.0)
    {
        // av_register_all deprecated in latest versions
        #if ( LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,9,100) )
        av_register_all();
        #endif

        if (Decode(fn, io_type) == -1) {
            avformat_close_input(&pFormatContext);
            return;
        }
        avformat_close_input(&pFormatContext);
        // Calculate accurate duration
        duration = totalSamplesPerChannel / originalSampleRate;
        // Construct NDArray
        ToNDArray();
    }

    AudioReader::~AudioReader() {

    }

    NDArray AudioReader::GetNDArray() {
        return output;
    }

    int AudioReader::GetNumPaddingSamples() {
        return std::ceil(padding * targetSampleRate);
    }

    double AudioReader::GetDuration() {
        return duration;
    }

    int64_t AudioReader::GetNumSamplesPerChannel() {
        if (outputVector.size() <= 0) return 0;
        return outputVector[0].size();
    }

    int AudioReader::GetNumChannels() {
        return numChannels;
    }

    void AudioReader::GetInfo() {
        std::cout << "audio stream [" << audioStreamIndex << "]: " << std::endl
            << "start time: " << std::endl
            << padding << std::endl
            << "duration: " << std::endl
            << duration << std::endl
            << "original sample rate: " << std::endl
            << originalSampleRate << std::endl
            << "target sample rate: " << std::endl
            << targetSampleRate << std::endl
            << "number of channels: " << std::endl
            << numChannels << std::endl
            << "total original samples per channel: " << std::endl
            << totalSamplesPerChannel << std::endl
            << "total original samples: " << std::endl
            << totalSamplesPerChannel * numChannels << std::endl
            << "total resampled samples per channel: " << std::endl
            << totalConvertedSamplesPerChannel << std::endl
            << "total resampled samples: " << std::endl
            << totalConvertedSamplesPerChannel  * numChannels << std::endl;
    }

    int AudioReader::Decode(std::string fn, int io_type) {
        // Get format context
        pFormatContext = avformat_alloc_context();
        CHECK(pFormatContext != nullptr) << "Unable to alloc avformat context";
        // Open input
        int formatOpenRet = 1;

        if (io_type == kDevice) {
            LOG(FATAL) << "Not implemented";
            return -1;
        } else if (io_type == kRawBytes) {
            filename = "BytesIO";
            io_ctx_.reset(new ffmpeg::AVIOBytesContext(fn, AVIO_BUFFER_SIZE));
            pFormatContext->pb = io_ctx_->get_avio();
            if (!pFormatContext->pb) {
                LOG(FATAL) << "Unable to init AVIO from memory buffer";
                return -1;
            }
            formatOpenRet = avformat_open_input(&pFormatContext, NULL, NULL, NULL);
        } else if (io_type == kNormal) {
            formatOpenRet = avformat_open_input(&pFormatContext, fn.c_str(), NULL, NULL);
        } else {
            LOG(FATAL) << "Invalid io type: " << io_type;
            return -1;
        }
        if (formatOpenRet != 0) {
            char errstr[200];
            av_strerror(formatOpenRet, errstr, 200);
            if (io_type != kBytes) {
                LOG(FATAL) << "ERROR opening: " << fn.c_str() << ", " << errstr;
            } else {
                LOG(FATAL) << "ERROR opening " << fn.size() << " bytes, " << errstr;
            }
            return -1;
        }
        // Read stream
        avformat_find_stream_info(pFormatContext, NULL);

        for (auto i = 0; i < int(pFormatContext->nb_streams); i++) {
            // find the stream needed
            AVCodecParameters *tempCodecParameters = pFormatContext->streams[i]->codecpar;
            if (tempCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                timeBase = (double)pFormatContext->streams[i]->time_base.num / (double)pFormatContext->streams[i]->time_base.den;
                duration = (double)pFormatContext->streams[i]->duration * timeBase;
                pCodecParameters = tempCodecParameters;
                originalSampleRate = tempCodecParameters->sample_rate;
                if (targetSampleRate == -1) targetSampleRate = originalSampleRate;
                numChannels = tempCodecParameters->channels;
                break;
            }
        }
        if (audioStreamIndex == -1) {
            LOG(FATAL) << "Can't find audio stream";
            return -1;
        }

        // prepare codec
        auto pCodec = avcodec_find_decoder(pCodecParameters->codec_id);
        CHECK(pCodec != nullptr) << "ERROR Decoder not found. THe codec is not supported.";
        pCodecContext = avcodec_alloc_context3(pCodec);
        CHECK(pCodecContext != nullptr) << "ERROR Could not allocate a decoding context.";
        CHECK_GE(avcodec_parameters_to_context(pCodecContext, pCodecParameters), 0) << "ERROR Could not set context parameters.";
        int codecOpenRet = 1;
        codecOpenRet = avcodec_open2(pCodecContext, pCodec, NULL);
        if (codecOpenRet < 0) {
            char errstr[200];
            av_strerror(codecOpenRet, errstr, 200);
            avcodec_close(pCodecContext);
            avcodec_free_context(&pCodecContext);
            avformat_close_input(&pFormatContext);
            LOG(FATAL) << "ERROR open codec through avcodec_open2: " << errstr;
            return -1;
        }
        // https://www.bilibili.com/read/cv2680761/
        pCodecContext->pkt_timebase = pFormatContext->streams[audioStreamIndex]->time_base;
        // prepare packet and frame
        AVPacket *pPacket = av_packet_alloc();
        AVFrame *pFrame = av_frame_alloc();
        DecodePacket(pPacket, pCodecContext, pFrame, audioStreamIndex);

        return 0;
    }

    void AudioReader::DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex) {
        // initialize resample context
        InitSWR(pCodecContext);

        // Get the packet
        int pktRet = -1;
        while ((pktRet = av_read_frame(pFormatContext, pPacket)) != AVERROR_EOF) {
            if (pktRet != 0) {
                LOG(WARNING) << "ERROR Fail to get packet.";
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
                    break;
                }
            }
            // Packet sent successfully, dont need it anymore
            av_packet_unref(pPacket);
            // Receive the decoded frames
            int receiveRet = -1;
            while ((receiveRet = avcodec_receive_frame(pCodecContext, pFrame)) == 0) {
                // Handle received frames
                totalSamplesPerChannel += pFrame->nb_samples;
                HandleFrame(pCodecContext, pFrame);
            }
            if (receiveRet != AVERROR(EAGAIN)) {
                LOG(WARNING) << "ERROR Fail to receive frame.";
                break;
            }
        }
        // Drain the decoder
        DrainDecoder(pCodecContext, pFrame);
        // clean up
        av_frame_free(&pFrame);
        av_packet_free(&pPacket);
        avcodec_close(pCodecContext);
        swr_close(swr);
        swr_free(&swr);
        avcodec_free_context(&pCodecContext);
        avformat_close_input(&pFormatContext);
    }

    void AudioReader::HandleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame) {
        // Add padding if necessary
        if (padding == -1.0) {
            padding = 0.0;
            if ((pFrame->pts * timeBase) > 0) {
                padding = pFrame->pts * timeBase;
            }
        }
        int ret = 0;
        // allocate resample buffer
        float** outBuffer;
        int outLinesize = 0;
        int outNumChannels = av_get_channel_layout_nb_channels(mono ? AV_CH_LAYOUT_MONO : pFrame->channel_layout);
        numChannels = outNumChannels;
        int outNumSamples = av_rescale_rnd(pFrame->nb_samples,
                                           this->targetSampleRate, pFrame->sample_rate, AV_ROUND_UP);
        if ((ret = av_samples_alloc_array_and_samples((uint8_t***)&outBuffer, &outLinesize, outNumChannels, outNumSamples,
                                                      AV_SAMPLE_FMT_FLTP, 0)) < 0)
        {
            LOG(FATAL) << "ERROR Failed to allocate resample buffer";
        }
        int gotSamples = 0;
        gotSamples = swr_convert(this->swr, (uint8_t**)outBuffer, outNumSamples, (const uint8_t**)pFrame->extended_data, pFrame->nb_samples);
        totalConvertedSamplesPerChannel += gotSamples;
        CHECK_GE(gotSamples, 0) << "ERROR Failed to resample samples";
        SaveToVector(outBuffer, outNumChannels, gotSamples);
        while (gotSamples > 0) {
            // flush buffer
            gotSamples = swr_convert(this->swr, (uint8_t**)outBuffer, outNumSamples, NULL, 0);
            CHECK_GE(gotSamples, 0) << "ERROR Failed to flush resample buffer";
            totalConvertedSamplesPerChannel += gotSamples;
            SaveToVector(outBuffer, outNumChannels, gotSamples);
        }
        if (outBuffer) {
            av_freep(&outBuffer[0]);
        }
        av_freep(&outBuffer);
    }

    void AudioReader::DrainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame) {
        int ret = 0;
        ret = avcodec_send_packet(pCodecContext, NULL);
        if (ret != 0) {
            LOG(WARNING) << "Failed to send packet while draining";
            return;
        }
        int receiveRet = -1;
        while ((receiveRet = avcodec_receive_frame(pCodecContext, pFrame)) == 0) {
            // Handle received frames
            totalSamplesPerChannel += pFrame->nb_samples;
            HandleFrame(pCodecContext, pFrame);
        }
        if (receiveRet != AVERROR(EAGAIN) && receiveRet != AVERROR_EOF) {
            LOG(WARNING) << "ERROR Fail to receive frame.";
        }
    }

    void AudioReader::InitSWR(AVCodecContext *pCodecContext) {
        int ret = 0;
        // Set resample ctx
        this->swr = swr_alloc();
        if (!this->swr) {
            LOG(FATAL) << "ERROR Failed to allocate resample context";
        }
        if (pCodecContext->channel_layout == 0) {
            pCodecContext->channel_layout = av_get_default_channel_layout( pCodecContext->channels );
        }
        av_opt_set_channel_layout(this->swr, "in_channel_layout",  pCodecContext->channel_layout, 0);
        av_opt_set_channel_layout(this->swr, "out_channel_layout", mono ? AV_CH_LAYOUT_MONO : pCodecContext->channel_layout,  0);
        av_opt_set_int(this->swr, "in_sample_rate",     pCodecContext->sample_rate,                0);
        av_opt_set_int(this->swr, "out_sample_rate",    this->targetSampleRate,                0);
        av_opt_set_sample_fmt(this->swr, "in_sample_fmt",  pCodecContext->sample_fmt, 0);
        av_opt_set_sample_fmt(this->swr, "out_sample_fmt", AV_SAMPLE_FMT_FLTP,  0);
        if ((ret = swr_init(this->swr)) < 0) {
            LOG(FATAL) << "ERROR Failed to initialize resample context";
        }
    }

    void AudioReader::ToNDArray() {
        if (outputVector.empty()) return;
        // Create the big NDArray
        int totalNumSamplesPerChannel = outputVector[0].size();
        std::vector<int64_t> shape {numChannels, totalNumSamplesPerChannel};
        output = NDArray::Empty(shape, kFloat32, ctx);
        // Create NDArray for each channel
        std::vector<int64_t> channelShape {totalNumSamplesPerChannel};
        for (int c = 0; c < numChannels; c++) {
            uint64_t offset = c * totalNumSamplesPerChannel;
            NDArray channelOutput = NDArray::Empty(channelShape, kFloat32, ctx);
            channelOutput.CopyFrom(outputVector[c], channelShape);
            auto view = output.CreateOffsetView(channelShape, channelOutput.data_->dl_tensor.dtype, &offset);
            channelOutput.CopyTo(view);
        }
    }

    void AudioReader::SaveToVector(float **buffer, int numChannels, int numSamples) {
        if (outputVector.empty()) {
            outputVector = std::vector<std::vector<float>>(numChannels, std::vector<float>());
        }
        for (int c = 0; c < numChannels; c++) {
            for (int s = 0; s < numSamples; s++) {
                float val = buffer[c][s];
                outputVector[c].push_back(val);
            }
        }
    }
}
