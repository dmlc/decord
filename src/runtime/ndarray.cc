/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <dmlc/logging.h>
#include <decord/runtime/ndarray.h>
#include <decord/runtime/c_runtime_api.h>
#include <decord/runtime/device_api.h>
#include "runtime_base.h"

// deleter for arrays used by DLPack exporter
extern "C" void NDArrayDLPackDeleter(DLManagedTensor* tensor);

namespace decord {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(NDArray::Container* ptr) {
    using decord::runtime::NDArray;
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      decord::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of DECORD.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(NDArray::Container* ptr) {
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(std::vector<int64_t> shape,
                        DLDataType dtype,
                        DLContext ctx) {
    VerifyDataType(dtype);
    // critical zone
    NDArray::Container* data = new NDArray::Container();
    data->deleter = DefaultDeleter;
    NDArray ret(data);
    ret.data_ = data;
    // RAII now in effect
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup ctx
    data->dl_tensor.ctx = ctx;
    return ret;
  }
  // Implementation of API function
  static DLTensor* MoveAsDLTensor(NDArray arr) {
    DLTensor* tensor = const_cast<DLTensor*>(arr.operator->());
    CHECK(reinterpret_cast<DLTensor*>(arr.data_) == tensor);
    arr.data_ = nullptr;
    return tensor;
  }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    CHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = NDArrayDLPackDeleter;
    return ret;
  }
};

NDArray NDArray::CreateView(std::vector<int64_t> shape,
                            DLDataType dtype) {
  CHECK(data_ != nullptr);
  CHECK(data_->dl_tensor.strides == nullptr)
      << "Can only create view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, data_->dl_tensor.ctx);
  ret.data_->dl_tensor.byte_offset =
      this->data_->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->data_->dl_tensor);
  size_t view_size = GetDataSize(ret.data_->dl_tensor);
  CHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // increase ref count
  this->data_->IncRef();
  ret.data_->manager_ctx = this->data_;
  ret.data_->dl_tensor.data = this->data_->dl_tensor.data;
  return ret;
}

NDArray NDArray::CreateOffsetView(std::vector<int64_t> shape,
                                  DLDataType dtype, uint64_t* offset) {
  CHECK(data_ != nullptr);
  CHECK(data_->dl_tensor.strides == nullptr)
      << "Can only create offset view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, data_->dl_tensor.ctx);
  ret.data_->dl_tensor.byte_offset =
      this->data_->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->data_->dl_tensor);
  size_t view_size = GetDataSize(ret.data_->dl_tensor);
  CHECK_LE(view_size + (*offset), curr_size)
      << "Tries to create a view that has bigger memory than current one with offset: " << *offset;
  // increase ref count
  this->data_->IncRef();
  ret.data_->manager_ctx = this->data_;
  ret.data_->dl_tensor.data = static_cast<char*>(this->data_->dl_tensor.data) + (*offset);
  *offset += view_size;
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const {
  return Internal::ToDLPack(data_);
}

NDArray NDArray::Empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  // setup memory content
  size_t size = GetDataSize(ret.data_->dl_tensor);
  size_t alignment = GetDataAlignment(ret.data_->dl_tensor);
  ret.data_->dl_tensor.data =
      DeviceAPI::Get(ret->ctx)->AllocDataSpace(
          ret->ctx, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  data->deleter = Internal::DLPackDeleter;
  data->manager_ctx = tensor;
  data->dl_tensor = tensor->dl_tensor;
  data->shape_.assign(data->dl_tensor.shape, data->dl_tensor.shape + data->dl_tensor.ndim);
  return NDArray(data);
}

void NDArray::CopyFromTo(DLTensor* from,
                         DLTensor* to,
                         DECORDStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
    << "DECORDArrayCopyFromTo: The size must exactly match";

  CHECK(from->ctx.device_type == to->ctx.device_type
        || from->ctx.device_type == kDLCPU
        || to->ctx.device_type == kDLCPU)
    << "Can not copy across different ctx types directly";

  // Use the context that is *not* a cpu context to get the correct device
  // api manager.
  DECORDContext ctx = from->ctx.device_type != kDLCPU ? from->ctx : to->ctx;

  DeviceAPI::Get(ctx)->CopyDataFromTo(
    from->data, static_cast<size_t>(from->byte_offset),
    to->data, static_cast<size_t>(to->byte_offset),
    from_size, from->ctx, to->ctx, from->dtype, stream);
}

}  // namespace runtime
}  // namespace decord

using namespace decord::runtime;

void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
  static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
  delete tensor;
}

int DECORDArrayAlloc(const decord_index_t* shape,
                  int ndim,
                  int dtype_code,
                  int dtype_bits,
                  int dtype_lanes,
                  int device_type,
                  int device_id,
                  DECORDArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = NDArray::Internal::MoveAsDLTensor(
      NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, ctx));
  API_END();
}

int DECORDArrayFree(DECORDArrayHandle handle) {
  API_BEGIN();
  reinterpret_cast<NDArray::Container*>(handle)->DecRef();
  API_END();
}

int DECORDArrayCopyFromTo(DECORDArrayHandle from,
                       DECORDArrayHandle to,
                       DECORDStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int DECORDArrayFromDLPack(DLManagedTensor* from,
                       DECORDArrayHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveAsDLTensor(NDArray::FromDLPack(from));
  API_END();
}

int DECORDArrayToDLPack(DECORDArrayHandle from,
                     DLManagedTensor** out) {
  API_BEGIN();
  *out = NDArray::Internal::ToDLPack(reinterpret_cast<NDArray::Container*>(from));
  API_END();
}

void DECORDDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

int DECORDArrayCopyFromBytes(DECORDArrayHandle handle,
                          void* data,
                          size_t nbytes) {
  API_BEGIN();
  DECORDContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "DECORDArrayCopyFromBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      data, 0,
      handle->data, static_cast<size_t>(handle->byte_offset),
      nbytes, cpu_ctx, handle->ctx, handle->dtype, nullptr);
  API_END();
}

int DECORDArrayCopyToBytes(DECORDArrayHandle handle,
                        void* data,
                        size_t nbytes) {
  API_BEGIN();
  DECORDContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "DECORDArrayCopyToBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      handle->data, static_cast<size_t>(handle->byte_offset),
      data, 0,
      nbytes, handle->ctx, cpu_ctx, handle->dtype, nullptr);
  API_END();
}
