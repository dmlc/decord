/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_loader.cc
 * \brief FFmpeg video loader, implements VideoLoaderInterface
 */

#include "video_loader.h"

#include <sstream>

namespace decord {
namespace ffmpeg {

FFMPEGVideoLoader::FFMPEGVideoLoader(std::vector<std::string> filenames, 
                                     std::vector<int> shape, int interval, 
                                     int skip, int shuffle, int prefetch) 
    : readers_(), shape_(shape), intvl_(interval), skip_(skip), shuffle_(shuffle), 
    prefetch_(prefetch), visit_order_(), visit_bounds_(), curr_(0) {
    // Validate parameters
    intvl_ = std::max(0, intvl_);
    skip_ = std::max(0, skip_);
    shuffle_ = std::max(0, shuffle_);
    prefetch_ = std::max(0, prefetch_);

    if (shape.size() != 4) {
        std::stringstream ss("(");
        for (auto s : shape) {
            ss << s << ", ";
        }
        ss << (")");
        LOG(FATAL) << "Shape must be of dim 4, in [Batchsize, H, W, C], given " << ss.str();
    } 

    // Initialize readers
    CHECK_GE(filenames.size(), 1) << "At least one video is required for video loader!";

    for (std::string filename : filenames) {
       ReaderPtr ptr = std::make_shared<FFMPEGVideoReader>(filename, shape_[2], shape_[1]);
       auto key_indices = ptr->GetKeyIndicesVector();
       CHECK_GT(key_indices.size(), 0) << "Error getting key frame info from " << filename;
       auto frame_count = ptr->GetFrameCount(); 
       CHECK_GT(frame_count, 0) << "Error getting total frame from " << filename;
       readers_.emplace_back(Entry(ptr, key_indices, frame_count));
    }

    // initialize visiting order for frames
    int bs = shape_[0];
    std::size_t entry_index = 0;
    for (auto entry : readers_) {
        int64_t len = entry.frame_count;
        int64_t i = 0;
        int64_t bs_skip = bs * (1 + intvl_) + skip_;
        for (i = 0; i < len; i += bs_skip) {
            visit_order_.emplace_back(std::make_pair(entry_index, i));
        }
        CHECK_LT(i, len + bs_skip);
        visit_bounds_.emplace_back(visit_order_.size());
        ++entry_index;
    }
    Reset();
}

void FFMPEGVideoLoader::Reset() {
    curr_ = 0;

    // basic case: no shuffle at all, sequentially read frames in filename order
    if (shuffle_ == 0) return;
    if (shuffle_ == 1) {
        // shuffle files, reads sequentially in each video
        std::size_t begin = 0;
        for (auto end : visit_bounds_) {
            std::random_shuffle(visit_order_.begin() + begin, visit_order_.begin() + end);
            begin = end;
        }
        CHECK(visit_order_.begin() + begin == visit_order_.end());
    } else if (shuffle_ == 2) {
        // shuffle files and re-order frames in each video, reading batches can be slower than other shuffle mode
        std::random_shuffle(visit_order_.begin(), visit_order_.end());
    } else {
        LOG(FATAL) << "No shuffle mode: " << shuffle_ << " supported.";
    }
}

FFMPEGVideoLoader::~FFMPEGVideoLoader() {

}

bool FFMPEGVideoLoader::HasNext() const {
    return (curr_ < visit_order_.size() - 1);
}

runtime::NDArray FFMPEGVideoLoader::Next() {
    if (!HasNext()) return NDArray::Empty({}, kUInt8, kCPU);
    CHECK(curr_ < visit_order_.size());
    auto pair = visit_order_[curr_];
    std::vector<int64_t> indices;
    indices.resize(shape_[0]);
    std::size_t reader_idx = pair.first;
    int64_t frame_idx = pair.second;
    for (auto i = 0; i < indices.size(); ++i) {
        indices.emplace_back(frame_idx);
        frame_idx += intvl_;
    }
    auto batch = readers_[reader_idx].ptr->GetBatch(indices);
    ++curr_;
    return batch;
}

int64_t FFMPEGVideoLoader::Length() const {
    return visit_order_.size();
}

}  // namespace ffmpeg
}  // namespace decord