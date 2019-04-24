/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef DECORD_RUNTIME_RUNTIME_BASE_H_
#define DECORD_RUNTIME_RUNTIME_BASE_H_

#include <decord/runtime/c_runtime_api.h>
#include <stdexcept>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(std::runtime_error &_except_) { return DECORDAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(std::runtime_error &_except_) { Finalize; return DECORDAPIHandleException(_except_); } return 0; // NOLINT(*)

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int DECORDAPIHandleException(const std::runtime_error &e) {
  DECORDAPISetLastError(e.what());
  return -1;
}

#endif  // DECORD_RUNTIME_RUNTIME_BASE_H_
