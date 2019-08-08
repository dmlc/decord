/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file storage_pool.cc
 * \brief Simple pool for storage
 */

#include "storage_pool.h"

namespace decord {

NDArrayPool::NDArrayPool() : init_(false) {

}

NDArrayPool::NDArrayPool(std::size_t sz, std::vector<int64_t> shape, DLDataType dtype, DLContext ctx)
    : size_(sz), shape_(shape), dtype_(dtype), ctx_(ctx), init_(true) {
}

NDArrayPool::~NDArrayPool() {
    while (queue_.size() > 0) {
        auto arr = queue_.front();
        queue_.pop();
        arr.data_->manager_ctx = nullptr;
    }
}

runtime::NDArray NDArrayPool::Acquire() {
    CHECK(init_) << "NDArrayPool not initialized with shape and ctx";
    if (queue_.size() > 0) {
        auto arr = queue_.front();
        queue_.pop();
        return arr;
    } else {
        // Allocate
        auto arr = NDArray::Empty(shape_, dtype_, ctx_);
        arr.data_->manager_ctx = this;
        arr.data_->deleter = &NDArrayPool::Deleter;
        return arr;
    }
}

void NDArrayPool::Deleter(NDArray::Container* ptr) {
    if (!ptr) return;
    if (ptr->manager_ctx != nullptr) {
        auto pool = static_cast<NDArrayPool*>(ptr->manager_ctx);
        if (pool->size_ <= pool->queue_.size()) {
            decord::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
            delete ptr;
            ptr = nullptr;
        } else {
            static_cast<NDArrayPool*>(ptr->manager_ctx)->queue_.push(NDArray(ptr));
        }
    } else if (ptr->dl_tensor.data != nullptr) {
        decord::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
        delete ptr;
        ptr = nullptr;
    }
}

}  // namespace decord
