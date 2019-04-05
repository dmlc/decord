/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_interface.h
 * \brief Video interface
 */

#ifndef DECORD_VIDEO_INTERFACE_H_
#define DECORD_VIDEO_INTERFACE_H_

#include <string>
#include <memory>

#include "base.h"
#include "runtime/ndarray.h"

namespace decord {
typedef void* VideoReaderInterfaceHandle;

enum AccessType {
    kVideoReaderRandomAccess = 0U,
    kVideoReaderSequential = 1U,
};

enum InterpolationType {
    kInterpolationLinear = 0U,
    kInterpolationBicubic = 1U,
    kInterpolationArea = 2U,
};

struct Size {
    uint32_t width;
    uint32_t height;
};

struct FrameProperty {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    DLDataType dtype;
    InterpolationType itype;
};

class VideoReaderInterface;
typedef std::shared_ptr<VideoReaderInterface> VideoReaderPtr;

// Video Reader is an abtract class defining the reader interfaces
class VideoReaderInterface {
 public:
    /*! \brief open video, return true if success */
    // virtual bool Open(std::string& fname, AccessType acc_type, FrameProperty prop) = 0;
    /*! \brief query stream info in video */
    virtual unsigned int QueryStreams() const = 0;
    /*! \brief check if video file successfully opened */
    virtual void SetVideoStream(int stream_nb = -1) = 0;
    /*! \brief get the total frame count in current stream */
    virtual int64_t GetFrameCount() const = 0;
    /*! \brief read the next frame, return NDArray */
    virtual runtime::NDArray NextFrame() = 0;
    /*! \brief retrieve keyframe indices */
    virtual runtime::NDArray GetKeyIndices() = 0;
    /*! \brief destructor */
    virtual ~VideoReaderInterface() = default;

    // The following APIs have perf concerns, use with caucious
    /*! \brief seek to position, this will clear all buffer and queue */
    // virtual runtime::NDArray Seek(uint64_t pos) = 0;
    virtual bool Seek(int64_t pos) = 0;
    /*! \brief seek and read frame at position p */
    // virtual runtime::NDArray GetFrame(uint64_t pos) = 0;
};  // class VideoReader

VideoReaderPtr GetVideoReader(std::string fname, 
                              Decoder dec = Decoder::FFMPEG());

class VideoLoader {
    public:
        using NDArray = runtime::NDArray;
        virtual ~VideoLoader() = 0;
        virtual bool HasNext() = 0;
        virtual NDArray Next() = 0;

    protected:
        // For doc only
        VideoLoader(std::vector<std::string> filenames, 
                    std::vector<int> shape, int inner_interval, 
                    int outer_interval, bool shuffle, 
                    int prefetch_frames, int num_shards);

    private:
        std::vector<VideoReaderInterface> readers_;
        std::vector<int> shape_;
        int interval_;
        int skip_;
        bool shuffle_;
        int num_prefetch_;
};  // class VideoLoader

}  // namespace decord
#endif // DECORD_VIDEO_INTERFACE_H_