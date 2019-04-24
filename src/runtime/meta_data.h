/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef DECORD_RUNTIME_META_DATA_H_
#define DECORD_RUNTIME_META_DATA_H_

#include <dmlc/json.h>
#include <dmlc/io.h>
#include <decord/runtime/packed_func.h>
#include <string>
#include <vector>
#include "runtime_base.h"

namespace decord {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<DECORDType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(dmlc::JSONWriter *writer) const;
  void Load(dmlc::JSONReader *reader);
  void Save(dmlc::Stream *writer) const;
  bool Load(dmlc::Stream *reader);
};
}  // namespace runtime
}  // namespace decord

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::decord::runtime::FunctionInfo, true);
}  // namespace dmlc
#endif  // DECORD_RUNTIME_META_DATA_H_
