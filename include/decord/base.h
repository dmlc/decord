/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file base.h
 * \brief configuration of decord and basic data structure.
 */
#ifndef DECORD_BASE_H_
#define DECORD_BASE_H_

#include <cstdint>
#include <string>
#include <ostream>

#include <dlpack/dlpack.h>
#include <dmlc/logging.h>


namespace decord {

// common data types
static const DLDataType kUInt8 = { kDLUInt, 8U, 1U };
static const DLDataType kUInt16 = { kDLUInt, 16U, 1U };
static const DLDataType kFloat16 = { kDLFloat, 16U, 1U };
static const DLDataType kFloat32 = { kDLFloat, 32U, 1U };
static const DLDataType kInt64 = {kDLInt, 64U, 1U};

/*! \brief check if current date type equals another one */
inline bool operator== (const DLDataType &d1, const DLDataType &d2) {
  return (d1.bits == d2.bits && d1.code == d2.code && d1.lanes == d2.lanes);
}

static const DLContext kCPU = {kDLCPU, 0};
static const DLContext kGPU = {kDLGPU, 0};

/*! \brief performance flags */
int constexpr kCPUAlignment = 32;  // For video width alignment, not comply to this will result in sparse arrays

}  // namespace decord
#endif  // DECORD_BASE_H_
