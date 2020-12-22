/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file str_util.cc
 * \brief Minimum string manipulation util for runtime.
 */

#include "str_util.h"
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

std::string GetEnvironmentVariableOrDefault(const std::string& variable_name,
                                            const std::string& default_value)
{
    const char* value = getenv(variable_name.c_str());
    return value ? value : default_value;
}

int ParseIntOrFloat(const std::string& str, int64_t& ivalue, double& fvalue) {
    char* p = nullptr;
    auto i = std::strtol(str.data(), &p, 10);
    if (p == str.data() + str.size()) {
        ivalue = int64_t(i);
        return 0;
    }

    auto f = std::strtod(str.data(), &p);
    if (p == str.data() + str.size()) {
        fvalue = f;
        return 1;
    }

    return -1;
}

}  // namespace runtime
}  // namespace decord
