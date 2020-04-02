/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file logging.cc
 * \brief Logging utils for decord
 */

#include "ffmpeg/ffmpeg_common.h"

#include <decord/runtime/registry.h>

namespace decord {
namespace runtime {

DECORD_REGISTER_GLOBAL("logging._CAPI_SetLoggingLevel")
.set_body([] (DECORDArgs args, DECORDRetValue* rv) {
    int log_level = args[0];
    av_log_set_level(log_level);
  });

}  // namespace runtime
}  // namespace decord
