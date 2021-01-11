//
// Created by Yin, Weisu on 1/8/21.
//

#include "audio_stream.h"

namespace decord {

    AudioStream::AudioStream(int sr, int nc, int duration)
    : streamSampleRate(sr), streamNumChannels(nc), duration(duration) { }

    int32_t AudioStream::GetDuration() {
        return duration;
    }

    int64_t AudioStream::GetSampleRate() {
        return streamSampleRate;
    }
}