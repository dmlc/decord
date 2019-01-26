/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader implementations
 */

#include <decord/video_reader.h>

namespace decord {
bool VideoReader::open(std::string& filename) {
    
    filename_ = filename;
    is_open_ = true;
    return is_open_;
}
}  // namespace decord