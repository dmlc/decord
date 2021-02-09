//
// Created by Yin, Weisu on 1/15/21.
//

#include <decord/audio_interface.h>
#include <decord/base.h>


int main() {
    auto audioReader = decord::GetAudioReader("/Users/weisy/Developer/yinweisu/decord/examples/example.mp3", -1, decord::kCPU, 0, 0);
    return 0;
}