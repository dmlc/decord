/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_loader.h
 * \brief FFmpeg video loader, implements VideoLoaderInterface
 */

#ifndef DECORD_VIDEO_FFMPEG_VIDEO_LOADER_H_
#define DECORD_VIDEO_FFMPEG_VIDEO_LOADER_H_

#include <decord/video_interface.h>

#include <vector>

#include "video_reader.h"

namespace decord {
namespace ffmpeg {

class FFMPEGVideoLoader : public VideoLoaderInterface {
public:
        FFMPEGVideoLoader(std::vector<std::string> filenames, 
                          std::vector<int> shape, int interval, 
                          int skip, bool shuffle, 
                          int prefetch);
        ~FFMPEGVideoLoader() = 0;
        bool HasNext() = 0;
        NDArray Next() = 0;

    
    private:
        using ReaderPtr = std::shared_ptr<FFMPEGVideoReader>;
        struct Entry {
            ReaderPtr ptr;
            std::vector<int64_t> key_indices;
            int64_t frame_count;

            Entry(ReaderPtr p, std::vector<int64_t> keys, int64_t frames)
                : ptr(ptr), key_indices(keys), frame_count(frames) {}
        };
        std::vector<Entry> readers_;
        std::vector<int> shape_;
        int interval_;
        int skip_;
        bool shuffle_;
        int num_prefetch_;
};  // class FFMPEGVideoLoader

}  // namespace ffmpeg
}  // namespace decord

#endif  //  DECORD_VIDEO_FFMPEG_VIDEO_LOADER_H_