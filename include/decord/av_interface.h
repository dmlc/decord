//
// Created by Yin, Weisu on 1/7/21.
//

#ifndef DECORD_AV_INTERFACE_H
#define DECORD_AV_INTERFACE_H

#include "runtime/ndarray.h"

namespace decord {

    using NDArray = runtime::NDArray;

    struct AVContainer {
        NDArray audio;
        NDArray video;
    };

    class AVReaderInterface {

    public:
        virtual AVContainer GetBatch(std::vector<int64_t> indices, AVContainer buf) = 0;
        virtual ~AVReaderInterface() = default;

    };
}

#endif //DECORD_AV_INTERFACE_H
