from ..base import DECORDError
from libcpp.vector cimport vector
from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport pycapsule
from libc.stdint cimport int64_t, uint64_t, uint8_t, uint16_t
import ctypes

cdef enum DECORDTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kHandle = 3
    kNull = 4
    kDECORDType = 5
    kDECORDContext = 6
    kArrayHandle = 7
    kNodeHandle = 8
    kModuleHandle = 9
    kFuncHandle = 10
    kStr = 11
    kBytes = 12
    kNDArrayContainer = 13
    kExtBegin = 15

cdef extern from "decord/runtime/c_runtime_api.h":
    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLContext:
        int device_type
        int device_id

    ctypedef struct DLTensor:
        void* data
        DLContext ctx
        int ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor* self)

    ctypedef struct DECORDValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str
        DLDataType v_type
        DLContext v_ctx

ctypedef int64_t decord_index_t
ctypedef DLTensor* DLTensorHandle
ctypedef void* DECORDStreamHandle
ctypedef void* DECORDRetValueHandle
ctypedef void* DECORDFunctionHandle
ctypedef void* NodeHandle

ctypedef int (*DECORDPackedCFunc)(
    DECORDValue* args,
    int* type_codes,
    int num_args,
    DECORDRetValueHandle ret,
    void* resource_handle)

ctypedef void (*DECORDPackedCFuncFinalizer)(void* resource_handle)

cdef extern from "decord/runtime/c_runtime_api.h":
    void DECORDAPISetLastError(const char* msg)
    const char *DECORDGetLastError()
    int DECORDFuncCall(DECORDFunctionHandle func,
                    DECORDValue* arg_values,
                    int* type_codes,
                    int num_args,
                    DECORDValue* ret_val,
                    int* ret_type_code)
    int DECORDFuncFree(DECORDFunctionHandle func)
    int DECORDCFuncSetReturn(DECORDRetValueHandle ret,
                          DECORDValue* value,
                          int* type_code,
                          int num_ret)
    int DECORDFuncCreateFromCFunc(DECORDPackedCFunc func,
                               void* resource_handle,
                               DECORDPackedCFuncFinalizer fin,
                               DECORDFunctionHandle *out)
    int DECORDCbArgToReturn(DECORDValue* value, int code)
    int DECORDArrayAlloc(decord_index_t* shape,
                      decord_index_t ndim,
                      DLDataType dtype,
                      DLContext ctx,
                      DLTensorHandle* out)
    int DECORDArrayFree(DLTensorHandle handle)
    int DECORDArrayCopyFromTo(DLTensorHandle src,
                           DLTensorHandle to,
                           DECORDStreamHandle stream)
    int DECORDArrayFromDLPack(DLManagedTensor* arr_from,
                           DLTensorHandle* out)
    int DECORDArrayToDLPack(DLTensorHandle arr_from,
                         DLManagedTensor** out)
    void DECORDDLManagedTensorCallDeleter(DLManagedTensor* dltensor)

# (minjie): Node and class module are not used in DECORD.
#cdef extern from "decord/c_dsl_api.h":
#    int DECORDNodeFree(NodeHandle handle)
#    int DECORDNodeTypeKey2Index(const char* type_key,
#                             int* out_index)
#    int DECORDNodeGetTypeIndex(NodeHandle handle,
#                            int* out_index)
#    int DECORDNodeGetAttr(NodeHandle handle,
#                       const char* key,
#                       DECORDValue* out_value,
#                       int* out_type_code,
#                       int* out_success)

cdef inline py_str(const char* x):
    if PY_MAJOR_VERSION < 3:
        return x
    else:
        return x.decode("utf-8")


cdef inline c_str(pystr):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return pystr.encode("utf-8")


cdef inline CALL(int ret):
    if ret != 0:
        raise DECORDError(py_str(DECORDGetLastError()))


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
