#include <decord/video_interface.h>
#include <dmlc/logging.h>
// #include <dmlc/io.h>
// #include <gtest/gtest.h>


int main(int argc, const char **argv) {
    auto vr = decord::GetVideoReader("test.mp4");
    vr->QueryStreams();
    auto array = vr->NextFrame();
    while (1) {
        array = vr->NextFrame();
        if (!array.defined()) break;
    }
    
    // auto stm = dmlc::Stream::Create("debug.params", "w");
    // array.Save(stm);
    LOG(INFO) << "end";
    return 0;
}