/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file random_file_order_sampler.cc
 * \brief Randomly shuffle file order but not internal frame order
 */

#include "random_file_order_sampler.h"

#include <algorithm>

#include <dmlc/logging.h>

namespace decord {
namespace sampler {

RandomFileOrderSampler::RandomFileOrderSampler(std::vector<int64_t> lens, std::vector<int64_t> range, int bs, int interval, int skip)
    : bs_(bs), visit_idx_(0) {
    CHECK_GT(bs_, 0) << "Batch size cannot be smaller than 1.";
    CHECK(range.size() % 2 == 0) << "Range (begin, end) size incorrect, expected: " << lens.size() * 2;
    CHECK_EQ(lens.size(), range.size() / 2) << "Video reader size mismatch with range: " << lens.size() << " vs " << range.size() / 2;

    // return sample buffer
    samples_.resize(bs_);

    // records for each video's property
    // visit order for shuffling
    records_.reserve(lens.size());
    visit_order_.clear();
    for (size_t i = 0; i < lens.size(); ++i) {
        auto begin = range[i*2];
        auto end = range[i*2 + 1];
        if (end < 0) {
            // allow negative indices, e.g., -20 means total_frame - 20
            end = lens[i] - end;
        }
        int64_t bs_skip = bs * (1 + interval) - interval + skip;
        // how many batchsizes in this reader
        int64_t num_batches = (end + skip - begin) / bs_skip;
        visit_order_.insert(visit_order_.end(), num_batches, i);
        CHECK_GE(end, 0) << "Video{" << i << "} has range end smaller than 0: " << end;
        CHECK(begin < end) << "Video{" << i << "} has invalid begin and end config: " << begin << "->" << end;
        CHECK(end < lens[i]) << "Video{" << i <<"} has range end larger than # frames: " << lens[i];
        records_.emplace_back(ReaderRecord{begin, end, interval, skip, begin});
    }
}

void RandomFileOrderSampler::Reset() {
    // shuffle orders
    std::random_shuffle(visit_order_.begin(), visit_order_.end());
    // reset visit idx
    visit_idx_ = 0;
    // clear and reset status to begin indices
    for (auto& record : records_) {
        record.current = record.begin;
    }
}

bool RandomFileOrderSampler::HasNext() const {
    return visit_idx_ < visit_order_.size();
}

const Samples& RandomFileOrderSampler::Next() {
    CHECK(HasNext());
    CHECK(samples_.size() == static_cast<size_t>(bs_));
    auto next_reader = visit_order_[visit_idx_];
    auto& record = records_[next_reader];
    auto pos = record.current;
    int idx = 0;
    for (idx = 0; idx < bs_; ++idx) {
        CHECK(pos < record.end);
        samples_[idx].first = next_reader;
        samples_[idx].second = pos;
        pos += record.interval + 1;
    }
    record.current = pos - record.interval + record.skip;
    ++visit_idx_;
    return samples_;
}

size_t RandomFileOrderSampler::Size() const {
    return visit_order_.size();
}
}  // sampler
}  // decord
