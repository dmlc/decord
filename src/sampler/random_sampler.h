/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file random_sampler.h
 * \brief Fully random sampler, with random file order and access position
 */

#ifndef DECORD_SAMPLER_RANDOM_SAMPLER_H_
#define DECORD_SAMPLER_RANDOM_SAMPLER_H_

#include "sampler_interface.h"

namespace decord {
namespace sampler {

class RandomSampler : public SamplerInterface {
    public:
        RandomSampler(std::vector<int64_t> lens, std::vector<int64_t> range, int bs, int interval, int skip);
        ~RandomSampler() = default;
        void Reset();
        bool HasNext() const;
        const Samples& Next();
        size_t Size() const;

    private:
        size_t bs_;
        Samples samples_;
        size_t curr_;
        std::vector<Samples> visit_order_;

};  // class RandomSampler

}  // sampler
}  // decord

#endif  // DECORD_SAMPLER_RANDOM_SAMPLER_H_
