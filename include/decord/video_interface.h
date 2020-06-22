/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
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
typedef void* VideoLoaderInterfaceHandle;

enum IOType {
    kNormal = 0U,    // normal file or URL
    kDevice,         // device, e.g., camera
    kRawBytes,       // raw bytes, e.g., raw data read from python file like object
};

enum VideoLoaderShuffleType {
    kSequential = 0U,
    kShuffleVideoOrderOnly,
    kShuffleBoth,
    kShuffleInsideVideoOnly,
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
    /*! \brief get the current frame position in current stream */
    virtual int64_t GetCurrentPosition() const = 0;
    /*! \brief read the next frame, return NDArray */
    virtual runtime::NDArray NextFrame() = 0;
    /*! \brief retrieve keyframe indices */
    virtual runtime::NDArray GetKeyIndices() = 0;
    /*! \brief retrieve playback seconds by frame indices */
    virtual runtime::NDArray GetFramePTS() const = 0;
    /*! \brief read bulk frames, defined by indices */
    virtual runtime::NDArray GetBatch(std::vector<int64_t> indices, runtime::NDArray buf) = 0;
    /*! \brief skip certain frames without decoding */
    virtual void SkipFrames(int64_t num = 1) = 0;
    /*! \brief get average fps */
    virtual double GetAverageFPS() const = 0;
    /*! \brief get rotation */
    virtual double GetRotation() const = 0;
    /*! \brief destructor */
    virtual ~VideoReaderInterface() = default;

    // The following APIs have perf concerns, use with caucious
    /*! \brief seek to position, this will clear all buffer and queue */
    // virtual runtime::NDArray Seek(uint64_t pos) = 0;
    /**
     * \brief Seek to nearest keyframe before frame pos
     *
     * \param pos The frame position
     * \return true Success
     * \return false Failed
     */
    virtual bool Seek(int64_t pos) = 0;
    /**
     * \brief Seek accurately to given position
     *
     * \param pos Frame position
     * \return true Success
     * \return false Failed
     */
    virtual bool SeekAccurate(int64_t pos) = 0;
    /*! \brief seek and read frame at position p */
    // virtual runtime::NDArray GetFrame(uint64_t pos) = 0;
};  // class VideoReader


DECORD_DLL VideoReaderPtr GetVideoReader(std::string fname, DLContext ctx,
                                         int width=-1, int height=-1, int nb_thread=0,
                                         int io_type=kNormal);

/**
 * \brief Interface of VideoLoader, pure virtual class
 *
 */
class VideoLoaderInterface {
    public:
        using NDArray = runtime::NDArray;
        virtual ~VideoLoaderInterface() = default;
        virtual void Reset() = 0;
        virtual bool HasNext() const = 0;
        virtual void Next() = 0;
        virtual NDArray NextData() = 0;
        virtual NDArray NextIndices() = 0;
        virtual int64_t Length() const = 0;
};  // class VideoLoaderInterface

}  // namespace decord
#endif // DECORD_VIDEO_INTERFACE_H_
