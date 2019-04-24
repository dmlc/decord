/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef DECORD_RUNTIME_MODULE_UTIL_H_
#define DECORD_RUNTIME_MODULE_UTIL_H_

#include <decord/runtime/module.h>
#include <decord/runtime/c_runtime_api.h>
#include <decord/runtime/c_backend_api.h>
#include <vector>

extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);
}  // extern "C"

namespace decord {
namespace runtime {
/*!
 * \brief Wrap a BackendPackedCFunc to packed function.
 * \param faddr The function address
 * \param mptr The module pointer node.
 */
PackedFunc WrapPackedFunc(BackendPackedCFunc faddr, const std::shared_ptr<ModuleNode>& mptr);
/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param module_list The module list to append to
 */
void ImportModuleBlob(const char* mblob, std::vector<Module>* module_list);

/*!
 * \brief Utility to initialize conext function symbols during startup
 * \param flookup A symbol lookup function.
 * \tparam FLookup a function of signature string->void*
 */
template<typename FLookup>
void InitContextFunctions(FLookup flookup) {
  #define DECORD_INIT_CONTEXT_FUNC(FuncName)                     \
    if (auto *fp = reinterpret_cast<decltype(&FuncName)*>     \
      (flookup("__" #FuncName))) {                            \
      *fp = FuncName;                                         \
    }
  // Initialize the functions
  DECORD_INIT_CONTEXT_FUNC(DECORDFuncCall);
  DECORD_INIT_CONTEXT_FUNC(DECORDAPISetLastError);
  DECORD_INIT_CONTEXT_FUNC(DECORDBackendGetFuncFromEnv);
  DECORD_INIT_CONTEXT_FUNC(DECORDBackendAllocWorkspace);
  DECORD_INIT_CONTEXT_FUNC(DECORDBackendFreeWorkspace);
  DECORD_INIT_CONTEXT_FUNC(DECORDBackendParallelLaunch);
  DECORD_INIT_CONTEXT_FUNC(DECORDBackendParallelBarrier);

  #undef DECORD_INIT_CONTEXT_FUNC
}
}  // namespace runtime
}  // namespace decord
#endif   // DECORD_RUNTIME_MODULE_UTIL_H_
