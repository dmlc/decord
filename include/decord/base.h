/*!
 *  Copyright (c) 2019 by Contributors
 * \file base.h
 * \brief configuration of DECORD and basic data structure.
 */
#ifndef DECORD_BASE_H_
#define DECORD_BASE_H_

#include <cstdint>
#include <string>

/*! \brief major version */
#define DECORD_MAJOR 0
/*! \brief minor version */
#define DECORD_MINOR 1
/*! \brief patch version */
#define DECORD_PATCH 0
/*! \brief mxnet version */
#define DECORD_VERSION (DECORD_MAJOR*10000 + DECORD_MINOR*100 + DECORD_PATCH)
/*! \brief helper for making version number */
#define DECORD_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + patch)

namespace decord {

/*! . */
struct Backend {
  /*! \brief Type of backend support */
  enum BackendType {
    kFFMPEG = 0,
    kNVDEC = 1,
  };  // enum BackendType

  /*! \brief Backend type. */
  BackendType be_type;

  /*! \brief default constructor */
  Backend() : be_type(kFFMPEG) {}
  /*! \brief constructor */
  Backend(BackendType type) : be_type(type) {}
  /*! \brief default FFMPEG backend */
  inline static Backend FFMPEG();

  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Backend &b) const {
    return be_type == b.be_type;
  }
}; // struct Backend

struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = 0,
    kGPU = 1,
  };  // enum DeviceType

  /*! \brief the device type we are on */
  DeviceType dev_type;
  /*! \brief device id we are on */
  int32_t dev_id;

  /*! \brief default constructor */
  Context() : dev_type(kCPU), dev_id(0) {}

  /*!
   * \brief Get corresponding device mask
   * \return kCPU or kGPU
   */
  inline DeviceType dev_mask() const {
    return dev_type;
  }

  /*!
   * \brief Returns dev_id for kGPU, 0 otherwise
   */
  inline int real_dev_id() const {
    if (dev_type == kGPU) return dev_id;
    return 0;
  }

  /*!
   * \brief Comparator, used to enable Context as std::map key.
   * \param b another context to compare
   * \return compared result
   */
  inline bool operator<(const Context &b) const;
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_type == b.dev_type && dev_id == b.dev_id;
  }
  /*!
   * \brief check if current context not equals another one
   * \param b another context to compare
   * \return whether they are not the same
   */
  inline bool operator!=(const Context &b) const {
    return !(*this == b);
  }

  /*! \brief the maximal device type */
  static const int32_t kMaxDevType = 6;
  /*! \brief the maximal device index */
  static const int32_t kMaxDevID = 16;

  /*!
   * \brief Create a new context.
   * \param dev_type device type.
   * \param dev_id device id. -1 for current device.
   */
  inline static Context Create(DeviceType dev_type, int32_t dev_id = -1);
  /*! \return CPU Context */
  inline static Context CPU(int32_t dev_id = 0);
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context. -1 for current GPU.
   */
  inline static Context GPU(int32_t dev_id = -1);
  /*!
   * Get the number of GPUs available.
   * \return The number of GPUs that are available.
   */
  inline static int32_t GetGPUCount();
  /*!
   * \brief get the free and total available memory on a GPU
   * \param dev the GPU number to query
   * \param free_mem pointer to the uint64_t holding free GPU memory
   * \param total_mem pointer to the uint64_t holding total GPU memory
   * \return No return value
   */
  inline static void GetGPUMemoryInformation(int dev, uint64_t *free, uint64_t *total);
  /*!
   * Create a context from string of the format [cpu|gpu|cpu_pinned](n)
   * \param str the string pattern
   * \return Context
   */
  inline static Context FromString(const std::string& str);
}; // struct Context

};  // namespace decord
#endif  // DECORD_BASE_H_
