/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_loader.h
 * \brief FFmpeg video loader, implements VideoLoaderInterface
 */

#ifndef DECORD_VIDEO_VIDEO_LOADER_H_
#define DECORD_VIDEO_VIDEO_LOADER_H_

#include "video_reader.h"
#include "../sampler/sampler_interface.h"

#include <vector>

#include <decord/video_interface.h>

namespace decord {

enum ShuffleTypes {
    kNoShuffle = 0,
    kRandomFileOrderShuffle,
    kRandomShuffle,
    kRandomInFileShuffle,
    kSmartRandomShuffle,
};  // enum ShuffleTypes

class VideoLoader : public VideoLoaderInterface {
public:
        VideoLoader(std::vector<std::string> filenames, std::vector<DLContext> ctxs,
                          std::vector<int> shape, int interval,
                          int skip, int shuffle,
                          int prefetch);
        ~VideoLoader();
        void Reset();
        bool HasNext() const;
        int64_t Length() const;
        void Next();
        NDArray NextData();
        NDArray NextIndices();

    private:
        using ReaderPtr = std::shared_ptr<VideoReader>;
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
        char next_ready_;  // ready flag, use with 0xFE for data, 0xFD for label
        NDArray next_data_;
        std::vector<int64_t> next_indices_;
        sampler::SamplerPtr sampler_;
        // std::vector<std::pair<std::size_t, int64_t> > visit_order_;
        // std::vector<std::size_t> visit_bounds_;
        // std::vector<std::vector<std::pair<std::size_t, int64_t> > > visit_buffer_;
        // std::size_t curr_;
        std::vector<DLContext> ctxs_;
        NDArrayPool ndarray_pool_;
};  // class VideoLoader
}  // namespace decord

#endif  //  DECORD_VIDEO_VIDEO_LOADER_H_
