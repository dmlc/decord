/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file smart_sampler.h
 * \brief Smart sampler for faster video random access
 */

#ifndef DECORD_SAMPLER_SMART_SAMPLER_H_
#define DECORD_SAMPLER_SMART_SAMPLER_H_

#include "sampler_interface.h"

namespace decord {
namespace sampler {

class SmartSampler : SamplerInterface {
    public:
        SmartSampler(std::vector<int64_t> lens, int interval, int bs_skip);

    private:

};  // class SmartSampler
    
}  // sampler
}  // decord

#endif  // DECORD_SAMPLER_SMART_SAMPLER_H_