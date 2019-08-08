/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file str_util.h
 * \brief Minimum string manipulation util for runtime.
 */
#ifndef DECORD_RUNTIME_STR_UTIL_H_
#define DECORD_RUNTIME_STR_UTIL_H_

#include <string>
#include <vector>

namespace decord {
namespace runtime {

std::vector<std::string> SplitString(std::string const &in, char sep);

}  // namespace decord
}  // namespace decord
#endif  // DECORD_RUNTIME_STR_UTIL_H_
