#include <decord/video_interface.h>
#include <decord/base.h>
#include <dmlc/logging.h>
#include <chrono>
#include <vector>
#include <algorithm>
// #include <dmlc/io.h>
// #include <gtest/gtest.h>

using NDArray = decord::runtime::NDArray;
using namespace decord;

std::time_t getTimeStamp() {
	std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
	return tp.time_since_epoch().count();
}

int main(int argc, const char **argv) {
    auto vr = decord::GetVideoReader("/tmp/testsrc_h264_10s_default.mp4", kCPU);
    LOG(INFO) << "Frame count: " << vr->GetFrameCount();
    vr->QueryStreams();
    NDArray array;
    int cnt = 0;
	auto start = getTimeStamp();
    while (0) {
        array = vr->NextFrame();
        // LOG(INFO) << array.Size();
        if (!array.Size()) break;
        cnt++;
        // if (cnt > 200) break;
        // LOG(INFO) << "Frame: " << cnt;
    }
    auto end = getTimeStamp();
	LOG(INFO) << cnt << " frame. Elapsed time: " << (end - start) / 1000.0;

	std::vector<int> indices(vr->GetFrameCount());
	std::iota(std::begin(indices), std::end(indices), 0);
	std::random_shuffle(std::begin(indices), std::end(indices));

	start = getTimeStamp();
	cnt = 0;
	for (size_t i = 0; i < 300; ++i) {
		if (i >= indices.size()) break;
		vr->SeekAccurate(indices[i]);
		array = vr->NextFrame();
		cnt++;
	}
	end = getTimeStamp();
	LOG(INFO) << cnt << " frame. Elapsed time: " << (end - start) / 1000.0;
    return 0;
}