/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file ndarray_pool.h
 * \brief Simple pool for ndarray
 */

#ifndef DECORD_UTILS_NDARRAY_POOL_H_
#define DECORD_UTILS_NDARRAY_POOL_H

#include <queue>
#include <vector>

#include <decord/runtime/ndarray.h>

namespace decord {

class NDArrayPool {
    using NDArray = runtime::NDArray;

    public:
        NDArrayPool(int pool_size, std::vector<int64_t> dshape, DLDataType dtype, DLContext ctx);
        NDArray Acquire();
        bool Recycle(NDArray arr);

    private:
        int pool_size_;
        std::vector<int64_t> dshape_;
        DLDataType dtype_;
        DLContext ctx_;
        std::queue<NDArray> queue_;

};  // class NDArrayPool

NDArrayPool::NDArrayPool(int pool_size, std::vector<int64_t> dshape, DLDataType dtype, DLContext ctx) 
    : pool_size_(pool_size), dshape_(dshape), dtype_(dtype), ctx_(ctx) {
}

runtime::NDArray NDArrayPool::Acquire() {
    if (queue_.empty()) {
        return NDArray::Empty(dshape_, dtype_, ctx_);
    } else {
        auto arr = queue_.front();
        queue_.pop();
        return arr;
    }
}

}  // namespace decord 

#endif  // DECORD_UTILS_NDARRAY_POOL_H_