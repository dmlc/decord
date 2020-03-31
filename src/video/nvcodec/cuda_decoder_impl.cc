/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder_impl.cc
 * \brief NVCUVID based decoder implementation details
 */

#include "cuda_decoder_impl.h"
#include "../../runtime/cuda/cuda_common.h"
#include <ostream>

namespace decord {
namespace cuda {
using namespace runtime;

const char * GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char *name;
    } aCodecName [] = {
        { cudaVideoCodec_MPEG1,    "MPEG-1"       },
        { cudaVideoCodec_MPEG2,    "MPEG-2"       },
        { cudaVideoCodec_MPEG4,    "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,      "VC-1/WMV"     },
        { cudaVideoCodec_H264,     "AVC/H.264"    },
        { cudaVideoCodec_JPEG,     "M-JPEG"       },
        { cudaVideoCodec_H264_SVC, "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC, "H.264/MVC"    },
        { cudaVideoCodec_HEVC,     "H.265/HEVC"   },
        { cudaVideoCodec_VP8,      "VP8"          }, // ?
        { cudaVideoCodec_VP9,      "VP9"          },
        { cudaVideoCodec_NumCodecs,"Invalid"      },
        { cudaVideoCodec_YUV420,   "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,     "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,     "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,     "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,     "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (size_t i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
        if (eCodec == aCodecName[i].eCodec) {
            return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}

const char * GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char *name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

CUVideoDecoderImpl::CUVideoDecoderImpl() : decoder_{0}, decoder_info_{}, initialized_{false} {
}

CUVideoDecoderImpl::CUVideoDecoderImpl(CUvideodecoder decoder)
    : decoder_{decoder}, decoder_info_{}, initialized_{true} {
}

CUVideoDecoderImpl::~CUVideoDecoderImpl() {
    if (initialized_) {
        CHECK_CUDA_CALL(cuvidDestroyDecoder(decoder_));
    }
}

CUVideoDecoderImpl::CUVideoDecoderImpl(CUVideoDecoderImpl&& other)
    : decoder_{other.decoder_}, initialized_{other.initialized_} {
    other.decoder_ = 0;
    other.initialized_ = false;
}

CUVideoDecoderImpl& CUVideoDecoderImpl::operator=(CUVideoDecoderImpl&& other) {
    if (initialized_) {
        CHECK_CUDA_CALL(cuvidDestroyDecoder(decoder_));
    }
    decoder_ = other.decoder_;
    initialized_ = other.initialized_;
    other.decoder_ = 0;
    other.initialized_ = false;
    return *this;
}

int CUVideoDecoderImpl::Initialize(CUVIDEOFORMAT* format) {
    if (initialized_) {
        if ((format->codec != decoder_info_.CodecType) ||
            (format->coded_width != decoder_info_.ulWidth) ||
            (format->coded_height != decoder_info_.ulHeight) ||
            (format->chroma_format != decoder_info_.ChromaFormat)) {
            std::cerr << "Encountered a dynamic video format change.\n";
            return 0;
        }
        return 1;
    }


    // DLOG(INFO) << "Hardware Decoder Input Information" << std::endl
    //             << "\tVideo codec     : " << GetVideoCodecString(format->codec) << std::endl
    //             << "\tFrame rate      : " << format->frame_rate.numerator << "/" << format->frame_rate.denominator
    //             << " = " << 1.0 * format->frame_rate.numerator / format->frame_rate.denominator << " fps" << std::endl
    //             << "\tSequence format : " << (format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
    //             << "\tCoded frame size: [" << format->coded_width << ", " << format->coded_height << "]" << std::endl
    //             << "\tDisplay area    : [" << format->display_area.left << ", " << format->display_area.top << ", "
    //             << format->display_area.right << ", " << format->display_area.bottom << "]" << std::endl
    //             << "\tChroma format   : " << GetVideoChromaFormatString(format->chroma_format) << std::endl
    //             << "\tBit depth       : " << format->bit_depth_luma_minus8 + 8 << std::endl;


    auto caps = CUVIDDECODECAPS{};
    caps.eCodecType = format->codec;
    caps.eChromaFormat = format->chroma_format;
    caps.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    if (CHECK_CUDA_CALL(cuvidGetDecoderCaps(&caps))) {
        if (!caps.bIsSupported) {
            std::stringstream ss;
            ss << "Unsupported Codec " << GetVideoCodecString(format->codec)
               << " with chroma format "
               << GetVideoChromaFormatString(format->chroma_format);
            LOG(FATAL) << ss.str();
        }
        // DLOG(INFO) << "NVDEC Capabilities" << std::endl
        //            << "\tMax width : " << caps.nMaxWidth << std::endl
        //            << "\tMax height : " << caps.nMaxHeight << std::endl
        //            << "\tMax MB count : " << caps.nMaxMBCount << std::endl
        //            << "\tMin width : " << caps.nMinWidth << std::endl
        //            << "\tMin height :" << caps.nMinHeight << std::endl;

        if (format->coded_width < caps.nMinWidth ||
            format->coded_height < caps.nMinHeight) {
            LOG(FATAL) << "Video is too small in at least one dimension.";
        }
        if (format->coded_width > caps.nMaxWidth ||
            format->coded_height > caps.nMaxHeight) {
            LOG(FATAL) << "Video is too large in at least one dimension.";
        }
        if (format->coded_width * format->coded_height / 256 > caps.nMaxMBCount) {
            LOG(FATAL) << "Video is too large (too many macroblocks).";
        }
    }

    decoder_info_.CodecType = format->codec;
    decoder_info_.ulWidth = format->coded_width;
    decoder_info_.ulHeight = format->coded_height;
    decoder_info_.ulNumDecodeSurfaces = 20;
    decoder_info_.ChromaFormat = format->chroma_format;
    decoder_info_.OutputFormat = cudaVideoSurfaceFormat_NV12;
    decoder_info_.bitDepthMinus8 = format->bit_depth_luma_minus8; // in ffmpeg but not sample
    decoder_info_.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    decoder_info_.ulTargetWidth = format->display_area.right - format->display_area.left;
    decoder_info_.ulTargetHeight = format->display_area.bottom - format->display_area.top;

    auto& area = decoder_info_.display_area;
    area.left   = format->display_area.left;
    area.right  = format->display_area.right;
    area.top    = format->display_area.top;
    area.bottom = format->display_area.bottom;
    // DLOG(INFO) << "\tUsing full size : [" << area.left << ", " << area.top
    //            << "], [" << area.right << ", " << area.bottom << "]" << std::endl;
    decoder_info_.ulNumOutputSurfaces = 2;
    decoder_info_.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    decoder_info_.vidLock = nullptr;

    if (CHECK_CUDA_CALL(cuvidCreateDecoder(&decoder_, &decoder_info_))) {
        initialized_ = true;
    } else {
        LOG(FATAL) << "Problem creating video decoder";
    }
    return 1;
}

bool CUVideoDecoderImpl::Initialized() const {
    return initialized_;
}

CUVideoDecoderImpl::operator CUvideodecoder() const {
    return decoder_;
}

uint16_t CUVideoDecoderImpl::Width() const {
    return static_cast<uint16_t>(decoder_info_.ulTargetWidth);
}

uint16_t CUVideoDecoderImpl::Height() const {
    return static_cast<uint16_t>(decoder_info_.ulTargetHeight);
}

}  // namespace cuda
}  // namespace decord
