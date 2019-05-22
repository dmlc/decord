#include <decord/video_interface.h>
#include <decord/base.h>
#include <dmlc/logging.h>
#include <chrono>
// #include <dmlc/io.h>
// #include <gtest/gtest.h>

using NDArray = decord::runtime::NDArray;
using namespace decord;

std::time_t getTimeStamp() {
	std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
	return tp.time_since_epoch().count();
}

int main(int argc, const char **argv) {
    auto vr = decord::GetVideoReader("test2.mp4", kGPU);
    LOG(INFO) << "Frame count: " << vr->GetFrameCount();
    vr->QueryStreams();
    NDArray array;
    int cnt = 0;
	auto start = getTimeStamp();
    while (1) {
        array = vr->NextFrame();
        // LOG(INFO) << array.Size();
        if (!array.Size()) break;
        cnt++;
        // if (cnt > 200) break;
        // LOG(INFO) << "Frame: " << cnt;
    }
    auto end = getTimeStamp();
	LOG(INFO) << cnt << " frame. Elapsed time: " << (end - start) / 1000.0;
    return 0;

    LOG(INFO) << " reset by seek";
    vr->SeekAccurate(3065);
    cnt = 0;
    while (1) {
        array = vr->NextFrame();
        // LOG(INFO) << array.Size();
        if (!array.Size()) break;
        cnt++;
        LOG(INFO) << "Frame: " << cnt;
    } 
    
    // auto stm = dmlc::Stream::Create("debug.params", "w");
    // array.Save(stm);
    LOG(INFO) << "end";
    return 0;
}