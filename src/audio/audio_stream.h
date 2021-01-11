//
// Created by Yin, Weisu on 1/8/21.
//

#ifndef DECORD_AUDIO_STREAM_H
#define DECORD_AUDIO_STREAM_H

#include "../video/ffmpeg/ffmpeg_common.h"

namespace decord {
    using AVFrames = std::vector<AVFrame>;

    class AudioStream {
    public:
        AudioStream(int sr, int nc, int duration);
        ~AudioStream();
        int64_t GetSampleRate();
        int32_t GetDuration();

        AVFrames audioFrames;
        int streamSampleRate;
        int streamNumChannels;
        int64_t duration;

    };
}

#endif //DECORD_AUDIO_STREAM_H
