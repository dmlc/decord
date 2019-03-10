/*!
 *  Copyright (c) 2019 by Contributors
 * \file video_interface.cc
 * \brief Video file reader implementations
 */

#include "./ffmpeg/video_reader.h"

#include <decord/video_interface.h>
#include <decord/runtime/registry.h>


#include <dmlc/logging.h>

namespace decord {

VideoReaderPtr GetVideoReader(std::string fn, Decoder be) {
    std::shared_ptr<VideoReaderInterface> ptr;
    if (be == Decoder::FFMPEG()) {
        // ptr = std::shared_ptr<VideoReaderInterface>(new ffmpeg::FFMPEGVideoReader(fn));
        ptr = std::make_shared<ffmpeg::FFMPEGVideoReader>(fn);
    } else {
        LOG(FATAL) << "Not supported Decoder type " << be;
    }
    return ptr;
}

namespace runtime {
DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoGetVideoReader")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    std::string fn = args[0];
    int width = args[1];
    int height = args[2];
    VideoReaderInterfaceHandle handle = static_cast<VideoReaderInterfaceHandle>(new ffmpeg::FFMPEGVideoReader(fn, width, height));
    *rv = handle;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoNextFrame")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    NDArray arr = static_cast<VideoReaderInterface*>(handle)->NextFrame();
    *rv = arr;
  });
}  // namespace runtime
}  // namespace decord