#include <decord/video_reader.h>
#include <dmlc/logging.h>
// #include <gtest/gtest.h>


int main(int argc, const char **argv) {
    auto vr = decord::GetVideoReader("test.mp4");
    vr->QueryStreams();
    auto array = vr->NextFrame();
    LOG(INFO) << "end";
    return 0;
}