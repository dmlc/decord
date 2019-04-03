#include <decord/video_interface.h>
#include <dmlc/logging.h>
// #include <dmlc/io.h>
// #include <gtest/gtest.h>

using NDArray = decord::runtime::NDArray;
int main(int argc, const char **argv) {
    auto vr = decord::GetVideoReader("test2.mp4");
    LOG(INFO) << "Frame count: " << vr->FrameCount();
    vr->QueryStreams();
    NDArray array = vr->NextFrame();
    while (1) {
        array = vr->NextFrame();
        // LOG(INFO) << array.Size();
        if (!array.Size()) break;
    }
    
    // auto stm = dmlc::Stream::Create("debug.params", "w");
    // array.Save(stm);
    LOG(INFO) << "end";
    return 0;
}