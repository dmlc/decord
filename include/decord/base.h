/*!
 *  Copyright (c) 2019 by Contributors
 * \file base.h
 * \brief configuration of DECORD and basic data structure.
 */
#ifndef DECORD_BASE_H_
#define DECORD_BASE_H_

/*! \brief major version */
#define DECORD_MAJOR 1
/*! \brief minor version */
#define DECORD_MINOR 5
/*! \brief patch version */
#define DECORD_PATCH 0
/*! \brief mxnet version */
#define DECORD_VERSION (DECORD_MAJOR*10000 + DECORD_MINOR*100 + DECORD_PATCH)
/*! \brief helper for making version number */
#define DECORD_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + patch)

namespace decord {

struct Handle {
  /*! \brief Type of backend support */
  enum BackendType {
    kFFMPEG,
    kNVDEC,
  };  // struct Handle

  /*! \brief Pointer to specific backend handle, type agnostic */
  void *ptr;
};

}  // namespace decord
#endif  // DECORD_BASE_H_
