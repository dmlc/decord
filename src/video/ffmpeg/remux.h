/*!
 *  Copyright (c) 2020 by Contributors if not otherwise specified
 * \file remux.h
 * \brief FFmpeg remux util function
 */

#ifndef DECORD_VIDEO_FFMPEG_REMUX_H_
#define DECORD_VIDEO_FFMPEG_REMUX_H_

#include "ffmpeg_common.h"

#include <string>

namespace decord {
namespace ffmpeg {

int InMemoryRemux(ffmpeg::AVFormatContextPtr fmt_ctx, std::string& out_filename);

}  // namespace ffmpeg
}  // namespace decord
#endif  // DECORD_VIDEO_FFMPEG_REMUX_H_
