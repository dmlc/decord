/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_loader.cc
 * \brief FFmpeg video loader, implements VideoLoaderInterface
 */

#include "video_loader.h"

#include <sstream>
#include <algorithm>

namespace decord {
namespace ffmpeg {

FFMPEGVideoLoader::FFMPEGVideoLoader(std::vector<std::string> filenames, 
                                     std::vector<int> shape, int interval, 
                                     int skip, int shuffle, int prefetch) 
    : readers_(), shape_(shape), intvl_(interval), skip_(skip), shuffle_(shuffle), 
    prefetch_(prefetch), visit_order_(), visit_bounds_(), visit_buffer_(), curr_(0) {
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
        int64_t bs_skip = bs * (1 + intvl_) - intvl_ + skip_;
        int64_t last = len - (bs * (1 + intvl_) - intvl_);
        for (i = 0; i < last; i += bs_skip) {
            visit_order_.emplace_back(std::make_pair(entry_index, i));
        }
        CHECK_LT(i, len + bs_skip);
        visit_bounds_.emplace_back(visit_order_.size());
        ++entry_index;
    }
    if (shuffle_ == 1) {
        visit_buffer_.reserve(readers_.size());
        std::size_t start = 0;
        for (auto bound : visit_bounds_) {
            visit_buffer_.emplace_back(std::vector<std::pair<std::size_t, int64_t> >(
                visit_order_.begin() + start, visit_order_.begin() + bound)); 
            start = bound;
        }
    }
    Reset();
}

void FFMPEGVideoLoader::Reset() {
    curr_ = 0;

    // basic case: no shuffle at all, sequentially read frames in filename order
    if (shuffle_ == 0) return;
    if (shuffle_ == 1) {
        // shuffle files order, reads sequentially in each video
        CHECK(visit_buffer_.size() == readers_.size());
        std::vector<std::size_t> random_array;
        random_array.reserve(visit_order_.size());
        for (std::size_t i = 0; i < visit_buffer_.size(); ++i) {
            for (std::size_t j = 0; j < visit_buffer_[i].size(); ++j) {
                random_array.emplace_back(i);
            }
        }
        std::random_shuffle(random_array.begin(), random_array.end());
        std::vector<std::pair<std::size_t, int64_t> > new_visit_order;
        std::vector<std::size_t> counters;
        counters.resize(visit_buffer_.size());
        new_visit_order.reserve(visit_order_.size());
        for (auto i : random_array) {
            new_visit_order.emplace_back(visit_buffer_[i][counters[i]]);
            ++counters[i];
        }
        CHECK_EQ(new_visit_order.size(), visit_order_.size());
        std::swap(new_visit_order, visit_order_);
    } else if (shuffle_ == 2) {
        // shuffle files and re-order frames in each video, reading batches can be slower than other shuffle mode
        std::random_shuffle(visit_order_.begin(), visit_order_.end());
    } else if (shuffle_ == 3) {
        // shuffle in video order, keeps original video file order, this should be rarely used.
        std::size_t begin = 0;
        for (auto end : visit_bounds_) {
            std::random_shuffle(visit_order_.begin() + begin, visit_order_.begin() + end);
            begin = end;
        }
        CHECK(visit_order_.begin() + begin == visit_order_.end());
    } else {
        LOG(FATAL) << "No shuffle mode: " << shuffle_ << " supported.";
    }
}

FFMPEGVideoLoader::~FFMPEGVideoLoader() {

}

bool FFMPEGVideoLoader::HasNext() const {
    return (curr_ < visit_order_.size());
}

runtime::NDArray FFMPEGVideoLoader::Next() {
    if (!HasNext()) return NDArray::Empty({}, kUInt8, kCPU);
    CHECK(curr_ < visit_order_.size());
    auto pair = visit_order_[curr_];
    std::vector<int64_t> indices;
    indices.reserve(shape_[0]);
    std::size_t reader_idx = pair.first;
    int64_t frame_idx = pair.second;
    for (auto i = 0; i < shape_[0]; ++i) {
        indices.emplace_back(frame_idx);
        frame_idx += intvl_ + 1;
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