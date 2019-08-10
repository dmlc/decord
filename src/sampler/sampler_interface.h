/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file sampler_interface.h
 * \brief Sampler interface
 */

#ifndef DECORD_SAMPLER_SAMPLER_INTERFACE_H_
#define DECORD_SAMPLER_SAMPLER_INTERFACE_H_

#include <memory>
#include <vector>
#include <utility>

namespace decord {
namespace sampler {

using SamplerIndex = std::pair<size_t, int64_t>;
using Samples = std::vector<SamplerIndex>;

class SamplerInterface {
    public:
        virtual ~SamplerInterface() = default;
        virtual void Reset() = 0;
        virtual bool HasNext() const = 0;
        virtual const Samples& Next() = 0;
        virtual size_t Size() const = 0;
};  // class SamplerInterface

using SamplerPtr = std::unique_ptr<SamplerInterface>;

}  // sampler
}  // decord
#endif
