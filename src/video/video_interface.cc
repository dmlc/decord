/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_interface.cc
 * \brief Video file reader implementations
 */

#include "video_reader.h"
#include "video_loader.h"
#include "../runtime/str_util.h"

#include <decord/video_interface.h>
#include <decord/runtime/registry.h>


#include <dlpack/dlpack.h>
#include <dmlc/logging.h>

namespace decord {

VideoReaderPtr GetVideoReader(std::string fn, DLContext ctx, int width, int height, int nb_thread, int io_type) {
    std::shared_ptr<VideoReaderInterface> ptr;
    ptr = std::make_shared<VideoReader>(fn, ctx, width, height, nb_thread, io_type);
    return ptr;
}

namespace runtime {
DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetVideoReader")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    std::string fn = args[0];
    int device_type = args[1];
    int device_id = args[2];
    int width = args[3];
    int height = args[4];
    int num_thread = args[5];
    int io_type = args[6];
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;
    auto reader = new VideoReader(fn, ctx, width, height, num_thread, io_type);
    if (reader->GetFrameCount() <= 0) {
      *rv = nullptr;
      return;
    }
    VideoReaderInterfaceHandle handle = static_cast<VideoReaderInterfaceHandle>(reader);
    *rv = handle;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderNextFrame")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    NDArray arr = static_cast<VideoReaderInterface*>(handle)->NextFrame();
    *rv = arr;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetFrameCount")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    int64_t ret = static_cast<VideoReaderInterface*>(handle)->GetFrameCount();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetCurrentPosition")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    int64_t ret = static_cast<VideoReaderInterface*>(handle)->GetCurrentPosition();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetKeyIndices")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    NDArray ret = static_cast<VideoReaderInterface*>(handle)->GetKeyIndices();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetFramePTS")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    NDArray ret = static_cast<VideoReaderInterface*>(handle)->GetFramePTS();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetBatch")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    NDArray indices = args[1];
    std::vector<int64_t> int_indices;
    indices.CopyTo(int_indices);
    NDArray ret = static_cast<VideoReaderInterface*>(handle)->GetBatch(int_indices, NDArray());
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderSeek")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    int64_t pos = args[1];
    bool ret = static_cast<VideoReaderInterface*>(handle)->Seek(pos);
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderSeekAccurate")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    int64_t pos = args[1];
    bool ret = static_cast<VideoReaderInterface*>(handle)->SeekAccurate(pos);
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderSkipFrames")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    int64_t num = args[1];
    static_cast<VideoReaderInterface*>(handle)->SkipFrames(num);
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderGetAverageFPS")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    double fps = static_cast<VideoReaderInterface*>(handle)->GetAverageFPS();
    *rv = fps;
  });

DECORD_REGISTER_GLOBAL("video_reader._CAPI_VideoReaderFree")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoReaderInterfaceHandle handle = args[0];
    auto p = static_cast<VideoReaderInterface*>(handle);
    if (p) delete p;
  });

// VideoLoader
DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderGetVideoLoader")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    CHECK_EQ(args.size(), 11);
    // for convenience, pass in comma separated filenames
    int idx = 0;
    std::string filenames = args[idx++];
    NDArray device_types = args[idx++];
    NDArray device_ids = args[idx++];
    int bs = args[idx++];
    int height = args[idx++];
    int width = args[idx++];
    int channel = args[idx++];
    int intvl = args[idx++];
    int skip = args[idx++];
    int shuffle = args[idx++];
    int prefetch = args[idx++];
    auto fns = SplitString(filenames, ',');
    std::vector<int> shape({bs, height, width, channel});
    // list of context
    std::vector<long int> dev_types;
    device_types.CopyTo(dev_types);
    std::vector<long int> dev_ids;
    device_ids.CopyTo(dev_ids);
    std::vector<DLContext> ctxs;
    ctxs.reserve(dev_ids.size());
    CHECK(dev_types.size() > 0);
    CHECK_EQ(dev_types.size(), dev_ids.size());
    for (std::size_t i = 0; i < dev_types.size(); ++i) {
      DLContext ctx;
      ctx.device_type = static_cast<DLDeviceType>(dev_types[i]);
      ctx.device_id = static_cast<int>(dev_ids[i]);
      ctxs.emplace_back(ctx);
    }
    VideoLoaderInterfaceHandle handle = static_cast<VideoLoaderInterfaceHandle>(new VideoLoader(fns, ctxs, shape, intvl, skip, shuffle, prefetch));
    *rv = handle;
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderReset")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    static_cast<VideoLoaderInterface*>(handle)->Reset();
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderLength")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    auto len = static_cast<VideoLoaderInterface*>(handle)->Length();
    *rv = len;
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderHasNext")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    bool ret = static_cast<VideoLoaderInterface*>(handle)->HasNext();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderNext")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    static_cast<VideoLoaderInterface*>(handle)->Next();
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderNextData")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    auto ret = static_cast<VideoLoaderInterface*>(handle)->NextData();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderNextIndices")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    auto ret = static_cast<VideoLoaderInterface*>(handle)->NextIndices();
    *rv = ret;
  });

DECORD_REGISTER_GLOBAL("video_loader._CAPI_VideoLoaderFree")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    VideoLoaderInterfaceHandle handle = args[0];
    auto p = static_cast<VideoLoaderInterface*>(handle);
    if (p) delete p;
  });
}  // namespace runtime
}  // namespace decord
