//
// Created by Yin, Weisu on 1/7/21.
//

#ifndef DECORD_AUDIO_INTERFACE_H
#define DECORD_AUDIO_INTERFACE_H

#include "runtime/ndarray.h"
#include "../../src/video/ffmpeg/ffmpeg_common.h"

namespace decord {
    using NDArray = runtime::NDArray;
    using NDArrays = std::vector<NDArray>;

    class AudioReaderInterface {
    public:
        virtual ~AudioReaderInterface() = default;
        virtual NDArrays GetBatch(std::vector<int> indices, NDArray buffer) = 0;

    };

}

#endif //DECORD_AUDIO_INTERFACE_H
