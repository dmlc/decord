//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AV_READER_H
#define DECORD_AV_READER_H

#include "../../include/decord/av_interface.h"
#include "../video/video_reader.h"
#include "../audio/audio_reader.h"

namespace decord {

    class AVReader {
        using VideoReaderPtr = std::unique_ptr<VideoReader>;
        using AudioReaderPtr = std::unique_ptr<AudioReader>;

    public:
        AVReader();
        ~AVReader();
    //    VideoReaderPtr GetVideoReader();
    //    AudioReaderPtr GetAudioReader();
        AVContainer GetBatch(std::vector<int64_t> indices, AVContainer buf);

        VideoReaderPtr video_reader;
        AudioReaderPtr audio_reader;

    private:
        void Sync();

    };

}
#endif //DECORD_AV_INTERFACE_H
