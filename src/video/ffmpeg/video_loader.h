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
                          int skip, int shuffle, 
                          int prefetch);
        ~FFMPEGVideoLoader();
        void Reset();
        bool HasNext() const;
        int64_t Length() const;
        NDArray Next();
    
    private:
        using ReaderPtr = std::shared_ptr<FFMPEGVideoReader>;
        struct Entry {
            ReaderPtr ptr;
            std::vector<int64_t> key_indices;
            int64_t frame_count;

            Entry(ReaderPtr p, std::vector<int64_t> keys, int64_t frames)
                : ptr(p), key_indices(keys), frame_count(frames) {}
        };
        std::vector<Entry> readers_;
        std::vector<int> shape_;
        int intvl_;
        int skip_;
        int shuffle_;
        int prefetch_;
        std::vector<std::pair<std::size_t, int64_t> > visit_order_;
        std::vector<std::size_t> visit_bounds_;
        std::size_t curr_;
};  // class FFMPEGVideoLoader

}  // namespace ffmpeg
}  // namespace decord

#endif  //  DECORD_VIDEO_FFMPEG_VIDEO_LOADER_H_