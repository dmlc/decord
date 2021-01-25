//
// Created by Yin, Weisu on 1/15/21.
//

#include <decord/audio_interface.h>
#include <decord/base.h>


int main() {
    auto audioReader = decord::GetAudioReader("/Users/weisy/Developer/yinweisu/decord/tests/cpp/audio/Kalimba.mp3", 44100, decord::kCPU);
    return 0;
}