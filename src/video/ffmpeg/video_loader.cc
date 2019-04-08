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
                                     int skip, bool shuffle, int prefetch) 
    : readers_({}), shape_(shape), interval_(interval), skip_(skip), shuffle_(shuffle), num_prefetch_(prefetch) {
    // Validate parameters
    interval_ = std::max(0, interval_);
    skip_ = std::max(0, skip_);
    num_prefetch_ = std::max(0, num_prefetch_);

    if (shape.size() != 4) {
        std::stringstream ss("(");
        for (auto s : shape) {
            ss << s << ", ";
        }
        ss << (")");
        LOG(FATAL) << "Shape must be of dim 4, in [Batchsize, C, H, W], given " << ss.str();
    } 

    // Initialize readers
    CHECK_GE(filenames.size(), 1) << "At least one video is required for video loader!";

    for (std::string filename : filenames) {
       ReaderPtr ptr = std::make_shared<FFMPEGVideoReader>(filename, shape_[3], shape_[2]);
       auto key_indices = ptr->GetKeyIndices();
       auto frame_count = ptr->GetFrameCount(); 
       readers_.emplace_back(Entry(ptr, key_indices, frame_count));
    }
}

}  // namespace ffmpeg
}  // namespace decord