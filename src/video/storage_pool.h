/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file ndarray_pool.h
 * \brief Simple pool for ndarray
 */

#ifndef DECORD_VIDEO_STORAGE_POOL_H_
#define DECORD_VIDEO_STORAGE_POOL_H_

#include <queue>
#include <vector>

#include <decord/runtime/ndarray.h>

namespace decord {

/**
 * \brief A pool with auto release memory management
 * 
 * \tparam T Pointer type
 * \tparam S Pool size
 */
template<typename T, int S>
class AutoReleasePool {
    public:
        using ptr_type = std::shared_ptr<T>;
        using pool_type = dmlc::ThreadLocalStore<std::queue<T*>>;
        /**
         * \brief Construct a new Auto Release Pool object
         * 
         */
        AutoReleasePool() : active_(true) {};
        /**
         * \brief Destroy the Auto Release Pool object
         * 
         */
        ~AutoReleasePool() {
            active_.store(false);
        }

        /**
         * \brief Acquire a new smart pointer object, either from pool (if exist) or from new memory
         * 
         * \return ptr_type 
         */
        ptr_type Acquire() {
            if (pool_type::Get()->empty()) {
                return std::shared_ptr<T>(Allocate(), std::bind(&AutoReleasePool::Recycle, this, std::placeholders::_1));
            }
            T* ret = pool_type::Get()->front();
            pool_type::Get()->pop();
            return std::shared_ptr<T>(ret, std::bind(&AutoReleasePool::Recycle, this, std::placeholders::_1));
        }

    private:
        /**
         * \brief Recycle function for on-destroying smart pointer object
         * 
         * \param p Raw pointer
         */
        void Recycle(T* p) {
            if (!p) return;
            if (!active_.load() || pool_type::Get()->size() + 1 > S) {
                Delete(p);
            } else {
                pool_type::Get()->push(p);
            }
        }

        /**
         * \brief Virtual allocation method for T*
         * 
         * \return T* New raw pointer
         */
        virtual T* Allocate() {
            LOG(FATAL) << "No entry";
            return new T;
        }

        /**
         * \brief Deleter for raw pointer
         * 
         * \param p Raw pointer to be freed
         */
        virtual void Delete(T* p) {
            LOG(FATAL) << "No entry";
            delete p;
        }

        /**
         * \brief whether pool is active or on-destroying
         * 
         */
        std::atomic<bool> active_;

    DISALLOW_COPY_AND_ASSIGN(AutoReleasePool);
};

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

#endif  // DECORD_VIDEO_STORAGE_POOL_H_