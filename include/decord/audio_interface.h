//
// Created by Yin, Weisu on 1/7/21.
//

#ifndef DECORD_AUDIO_INTERFACE_H
#define DECORD_AUDIO_INTERFACE_H

#include "runtime/ndarray.h"
#include "../../src/video/ffmpeg/ffmpeg_common.h"

namespace decord {
    typedef void* AudioReaderInterfaceHandle;

    class AudioReaderInterface;
    typedef std::shared_ptr<AudioReaderInterface> AudioReaderPtr;

    using NDArray = runtime::NDArray;

    class AudioReaderInterface {
    public:
        virtual ~AudioReaderInterface() = default;
        virtual NDArray GetNDArray() = 0;
        virtual int GetNumPaddingSamples() = 0;
        virtual double GetDuration() = 0;
        virtual int64_t GetNumSamplesPerChannel() = 0;
        virtual int GetNumChannels() = 0;
    };

    DECORD_DLL AudioReaderPtr GetAudioReader(std::string fname, int sampleRate, DLContext ctx);
}

#endif //DECORD_AUDIO_INTERFACE_H
