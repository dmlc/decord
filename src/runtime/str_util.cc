/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file str_util.cc
 * \brief Minimum string manipulation util for runtime.
 */

#include "file_util.h"

namespace decord {
namespace runtime {

std::vector<std::string> SplitString(std::string const &in, char sep) {
    std::string::size_type b = 0;
    std::vector<std::string> result;

    while ((b = in.find_first_not_of(sep, b)) != std::string::npos) {
        auto e = in.find_first_of(sep, b);
        result.push_back(in.substr(b, e-b));
        b = e;
    }
    return result;
}

}  // namespace runtime
}  // namespace decord
