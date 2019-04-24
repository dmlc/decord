/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file decord/runtime/util.h
 * \brief Useful runtime util.
 */
#ifndef DECORD_RUNTIME_UTIL_H_
#define DECORD_RUNTIME_UTIL_H_

#include "c_runtime_api.h"

namespace decord {
namespace runtime {

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DECORDType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
}  // namespace runtime
}  // namespace decord
// Forward declare the intrinsic id we need
// in structure fetch to enable stackvm in runtime
namespace decord {
namespace ir {
namespace intrinsic {
/*! \brief The kind of structure field info used in intrinsic */
enum DECORDStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // DECORDValue field
  kDECORDValueContent,
  kDECORDValueKindBound_
};
}  // namespace intrinsic
}  // namespace ir
}  // namespace decord
#endif  // DECORD_RUNTIME_UTIL_H_
