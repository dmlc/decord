/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file random_file_order_sampler.cc
 * \brief Randomly shuffle file order but not internal frame order
 */

#include "sequential_sampler.h"

#include <dmlc/logging.h>

namespace decord {
namespace sampler {

SequentialSampler::SequentialSampler(std::vector<int64_t> lens, std::vector<int64_t> range, int bs, int interval, int skip)
    : bs_(bs), curr_(0) {
    CHECK(range.size() % 2 == 0) << "Range (begin, end) size incorrect, expected: " << lens.size() * 2;
    CHECK_EQ(lens.size(), range.size() / 2) << "Video reader size mismatch with range: " << lens.size() << " vs " << range.size() / 2;

    // output buffer
    samples_.resize(bs);

    // visit order for shuffling
    visit_order_.clear();
    for (size_t i = 0; i < lens.size(); ++i) {
        auto begin = range[i*2];
        auto end = range[i*2 + 1];
        if (end < 0) {
            // allow negative indices, e.g., -20 means total_frame - 20
            end = lens[i] - end;
        }
        CHECK_GE(end, 0) << "Video{" << i << "} has range end smaller than 0: " << end;
        CHECK(begin < end) << "Video{" << i << "} has invalid begin and end config: " << begin << "->" << end;
        CHECK(end < lens[i]) << "Video{" << i <<"} has range end larger than # frames: " << lens[i];
        int64_t bs_skip = bs * (1 + interval) - interval + skip;
        int64_t bs_length = bs_skip - skip;
        for (int64_t b = begin; b + bs_length < end; b += bs_skip) {
            int offset = 0;
            for (int j = 0; j < bs; ++j) {
                samples_[j] = std::make_pair(i, b + offset);
                offset += interval + 1;
            }
            visit_order_.emplace_back(samples_);
        }
    }
}

void SequentialSampler::Reset() {
    // reset visit idx
    curr_ = 0;
}

bool SequentialSampler::HasNext() const {
    return curr_ < visit_order_.size();
}

const Samples& SequentialSampler::Next() {
    CHECK(HasNext());
    CHECK_EQ(samples_.size(), bs_);
    samples_ = visit_order_[curr_++];
    return samples_;
}

size_t SequentialSampler::Size() const {
    return visit_order_.size();
}
}  // sampler
}  // decord
