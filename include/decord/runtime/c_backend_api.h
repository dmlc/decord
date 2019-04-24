/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file decord/runtime/c_backend_api.h
 * \brief DECORD runtime backend API.
 *
 *  The functions defined in this header are intended to be
 *  used by compiled decord operators, usually user do not need to use these
 *  function directly.
 */
#ifndef DECORD_RUNTIME_C_BACKEND_API_H_
#define DECORD_RUNTIME_C_BACKEND_API_H_

#include "c_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend related functions.
/*!
 * \brief Backend function for modules to get function
 *  from its environment mod_node (its imports and global function).
 *  The user do should not call DECORDFuncFree on func.
 *
 * \param mod_node The module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDBackendGetFuncFromEnv(void* mod_node,
                                     const char* func_name,
                                     DECORDFunctionHandle *out);
/*!
 * \brief Backend function to register system-wide library symbol.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDBackendRegisterSystemLibSymbol(const char* name, void* ptr);

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocate spaced is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
DECORD_DLL void* DECORDBackendAllocWorkspace(int device_type,
                                       int device_id,
                                       uint64_t nbytes,
                                       int dtype_code_hint,
                                       int dtype_bits_hint);

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa DECORDBackendAllocWorkspace
 */
DECORD_DLL int DECORDBackendFreeWorkspace(int device_type,
                                    int device_id,
                                    void* ptr);

/*!
 * \brief Environment for DECORD parallel task.
 */
typedef struct {
  /*!
   * \brief Auxiliary used for synchronization
   */
  void* sync_handle;
  /*! \brief total amount of task */
  int32_t num_task;
} DECORDParallelGroupEnv;

/*!
 * \brief The callback function to execute a parallel lambda
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \param cdata The supporting closure data.
 */
typedef int (*FDECORDParallelLambda)(
    int task_id, DECORDParallelGroupEnv* penv, void* cdata);

/*!
 * \brief Backend function for running parallel jobs.
 *
 * \param flambda The parallel function to be launched.
 * \param cdata The closure data.
 * \param num_task Number of tasks to launch, can be 0, means launch
 *           with all available threads.
 *
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDBackendParallelLaunch(FDECORDParallelLambda flambda,
                                     void* cdata,
                                     int num_task);

/*!
 * \brief BSP barrrier between parallel threads
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDBackendParallelBarrier(int task_id, DECORDParallelGroupEnv* penv);


/*!
 * \brief Simple static initialization function.
 *  Run f once and set handle to be not null.
 *  This function is mainly used for test purpose.
 *
 * \param handle An global address to indicate f
 * \param f The function to be ran
 * \param cdata The closure data to pass to the function.
 * \param nbytes Number of bytes in the closure data.
 * \return 0 when no error is thrown, -1 when failure happens
 */
DECORD_DLL int DECORDBackendRunOnce(void** handle,
                              int (*f)(void*),
                              void *cdata,
                              int nbytes);

#ifdef __cplusplus
}  // DECORD_EXTERN_C
#endif
#endif  // DECORD_RUNTIME_C_BACKEND_API_H_
