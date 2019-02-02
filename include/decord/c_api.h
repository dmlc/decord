/*!
 *  Copyright (c) 2016 by Contributors
 * \file decord/c_api.h
 * \brief DECORD C APIs.
 *
 *  The C APIs are from a minimum set to support DECORD functionality in foreign languages, 
 *  such as python, JAVA.
 */
#ifndef DECORD_C_API_H_
#define DECORD_C_API_H_

// Macros to do weak linking
#ifdef _MSC_VER
#define DECORD_WEAK __declspec(selectany)
#else
#define DECORD_WEAK __attribute__((weak))
#endif

#ifndef DECORD_DLL
#ifdef _WIN32
#ifdef DECORD_EXPORTS
#define DECORD_DLL __declspec(dllexport)
#else
#define DECORD_DLL __declspec(dllimport)
#endif
#else
#define DECORD_DLL __attribute__((visibility("default")))
#endif
#endif

// DECORD version
#define DECORD_VERSION "0.0.1"

// DECORD is DLPack compatible.
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stddef.h>

/*! \brief type of array index. */
typedef int64_t dd_index_t;

/*!
 * \brief The type code in DECORDType
 * \note DECORDType is used in two places. 
 * To maintain maximum compatibility with MXNet/DECORD, this extension list is as similar as possible in DECORD
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by DECORD API calls.
  kHandle = 3U,
  kNull = 4U,
  kDECORDType = 5U,
  kDECORDContext = 6U,
  kArrayHandle = 7U,
  kNodeHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  // Extension codes for other frameworks to integrate DECORD PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U
} DECORDTypeCode;

/*!
 * \brief The data type used in DECORD Runtime.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *
 * \note Arguments DECORD API function always takes bits=64 and lanes=1
 */
typedef DLDataType DECORDType;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef DLContext DECORDContext;

/*!
 * \brief The tensor array stucture to DECORD API.
 */
typedef DLTensor DECORDArray;

/*! \brief the array handle */
typedef DECORDArray* DECORDArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DECORDType v_type;
  DECORDContext v_ctx;
} DECORDValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} DECORDByteArray;


/*! \brief Handle to hold return value. */
typedef void* DECORDRetValueHandle;

/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* DECORDStreamHandle;

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
DECORD_DLL void DECORDAPISetLastError(const char* msg);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  DECORDGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
DECORD_DLL const char *DECORDGetLastError(void);

/*!
 * \brief Free front-end extension type resource.
 * \param handle The extension handle.
 * \param type_code The type of of the extension type.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDExtTypeFree(void* handle, int type_code);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayAlloc(const dd_index_t* shape,
                                int ndim,
                                int dtype_code,
                                int dtype_bits,
                                int dtype_lanes,
                                int device_type,
                                int device_id,
                                DECORDArrayHandle* out);

/*!
 * \brief Free the DECORD Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayFree(DECORDArrayHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyFromBytes(DECORDArrayHandle handle,
                                        void* data,
                                        size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyToBytes(DECORDArrayHandle handle,
                                      void* data,
                                      size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayCopyFromTo(DECORDArrayHandle from,
                                     DECORDArrayHandle to,
                                     DECORDStreamHandle stream);

/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayFromDLPack(DLManagedTensor* from,
                                     DECORDArrayHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDArrayToDLPack(DECORDArrayHandle from,
                             DLManagedTensor** out);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
DECORD_DLL void DECORDDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/*!
 * \brief Create a new runtime stream.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param out The new stream handle
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamCreate(int device_type, int device_id, DECORDStreamHandle* out);

/*!
 * \brief Free a created stream handle.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param stream The stream to be freed
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamFree(int device_type, int device_id, DECORDStreamHandle stream);

/*!
 * \brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param handle The stream handle.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDSetStream(int device_type, int device_id, DECORDStreamHandle handle);

/*!
 * \brief Wait until all computations on stream completes.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param stream The stream to be synchronized.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDSynchronize(int device_type, int device_id, DECORDStreamHandle stream);

/*!
 * \brief Synchronize two streams of execution.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, -1 when failure happens
 */
DECORD_DLL int DECORDStreamStreamSynchronize(int device_type,
                                       int device_id,
                                       DECORDStreamHandle src,
                                       DECORDStreamHandle dst);

#ifdef __cplusplus
}  // DECORD_EXTERN_C
#endif
#endif  // DECORD_C_API_H_