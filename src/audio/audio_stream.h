//
// Created by Yin, Weisu on 1/8/21.
//

#ifndef DECORD_AUDIO_STREAM_H
#define DECORD_AUDIO_STREAM_H

#include "../video/ffmpeg/ffmpeg_common.h"

namespace decord {
    using NDArray = runtime::NDArray;

    class AudioStream {
    public:
        AudioStream(int sr, int nc, double duration);
        ~AudioStream();
        int64_t GetSampleRate();
        int32_t GetDuration();

        NDArray audioSamples;
        int streamSampleRate;
        int streamNumChannels;
        double duration;

    };
}

#endif //DECORD_AUDIO_STREAM_H
