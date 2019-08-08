/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file random_file_order_sampler.h
 * \brief Randomly shuffle file order but not internal frame order
 */

#ifndef DECORD_SAMPLER_RANDOM_FILE_ORDER_SAMPLER_H_
#define DECORD_SAMPLER_RANDOM_FILE_ORDER_SAMPLER_H_

#include "sampler_interface.h"

namespace decord {
namespace sampler {

class RandomFileOrderSampler : public SamplerInterface {
    public:
        RandomFileOrderSampler(std::vector<int64_t> lens, std::vector<int64_t> range, int bs, int interval, int skip);
        ~RandomFileOrderSampler() = default;
        void Reset();
        bool HasNext() const;
        const Samples& Next();
        size_t Size() const;

    private:
        struct ReaderRecord {
            // immutable
            const int64_t begin;
            const int64_t end;
            const int interval;
            const int skip;
            // mutable
            int64_t current;
        };  // struct Record

        int bs_;
        Samples samples_;
        std::vector<ReaderRecord> records_;
        std::vector<size_t> visit_order_;
        std::size_t visit_idx_;

};  // class RandomFileOrderSampler

}  // sampler
}  // decord

#endif  // DECORD_SAMPLER_RANDOM_FILE_ORDER_SAMPLER_H_
