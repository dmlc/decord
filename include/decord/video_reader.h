/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_reader.h
 * \brief Video file reader
 */

#ifndef DECORD_VIDEO_READER_H_
#define DECORD_VIDEO_READER_H_

#include "base.h"
#include "video_stream.h"
#include <string>

namespace decord {

class VideoReader {
 public:
    /*! \brief default constructor */
    VideoReader(std::string& filename) : backend_(Backend::FFMPEG()), is_open_(false) {
         this->open(filename); 
    }
    /*! \brief constructor with filename and backend type */
    VideoReader(std::string& filename, Backend::BackendType type) 
        : backend_(type), is_open_(false) { this->open(filename); }
    /*! \brief destructor */
    ~VideoReader() { this->close(); }
    /*! \brief check if video file successfully opened */
    bool is_open() { return is_open_; }

 private:
    /*! \brief video file name */
    std::string filename_;
    /*! \brief backend used to decode video */
    Backend backend_;
    /*! \brief Pointer to backend handle */
    void *hdl_ptr_;
    /*! \brief open status */
    bool is_open_;
    /*! \brief open file for read and decode */
    bool open(std::string& filename);
    /*! \brief close file */
    void close();
};  // struct VideoReader
}  // namespace decord
#endif // DECORD_VIDEO_READER_H_