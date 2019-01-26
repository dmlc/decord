/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader_impl.h
 * \brief Video file reader pure abstract impl class
 */

#ifndef DECORD_VIDEO_READER_IMPL_H_
#define DECORD_VIDEO_READER_IMPL_H_

#include <string>

namespace decord {

class VideoReaderImpl {
 public:
    /*!
    * \brief Open file for read.
    */
    virtual bool open(std::string& filename) = 0;
    /*!
    * \brief Close file.
    */
    virtual void close() = 0;
    /*!
    * \brief Destructor.
    */
    virtual ~VideoReaderImpl() = default;
};  // class VideoReaderImpl
}  // namespace decord

#endif  // DECORD_VIDEO_READER_IMPL_H_