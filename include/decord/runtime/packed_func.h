/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file decord/runtime/packed_func.h
 * \brief Type-erased function used across DECORD API.
 */
#ifndef DECORD_RUNTIME_PACKED_FUNC_H_
#define DECORD_RUNTIME_PACKED_FUNC_H_

#include <dmlc/logging.h>
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <type_traits>
#include "c_runtime_api.h"
#include "module.h"
#include "ndarray.h"
#include "node_base.h"

namespace HalideIR {
// Forward declare type for extensions
// The header works fine without depending on this.
struct Type;
struct Expr;
}

// Whether use DECORD runtime in header only mode.
#ifndef DECORD_RUNTIME_HEADER_ONLY
#define DECORD_RUNTIME_HEADER_ONLY 0
#endif

namespace decord {
// forward declarations
class Integer;

namespace runtime {
// forward declarations
class DECORDArgs;
class DECORDArgValue;
class DECORDRetValue;
class DECORDArgsSetter;

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of DECORD.
 *  It corresponds to DECORDFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /*!
   * \brief The internal std::function
   * \param args The arguments to the function.
   * \param rv The return value.
   *
   * \code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(DECORDArgs args, DECORDRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * \endcode
   */
  using FType = std::function<void (DECORDArgs args, DECORDRetValue* rv)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*! \brief constructor from null */
  PackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template<typename... Args>
  inline DECORDRetValue operator()(Args&& ...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  inline void CallPacked(DECORDArgs args, DECORDRetValue* rv) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return body_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return body_ != nullptr;
  }

 private:
  /*! \brief internal container of packed function */
  FType body_;
};

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor "TypedPackedFunc<R(Args..)>"
 */
template<typename FType>
class TypedPackedFunc;

/*!
 * \anchor TypedPackedFuncAnchor
 * \brief A PackedFunc wrapper to provide typed function signature.
 * It is backed by a PackedFunc internally.
 *
 * TypedPackedFunc enables compile time type checking.
 * TypedPackedFunc works with the runtime system:
 * - It can be passed as an argument of PackedFunc.
 * - It can be assigned to DECORDRetValue.
 * - It can be directly converted to a type-erased PackedFunc.
 *
 * Developers should prefer TypedPackedFunc over PackedFunc in C++ code
 * as it enables compile time checking.
 * We can construct a TypedPackedFunc from a lambda function
 * with the same signature.
 *
 * \code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedPackedFunc
 *  TypedPackedFunc<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to PackedFunc
 *  PackedFunc packed = ftype;
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template<typename R, typename ...Args>
class TypedPackedFunc<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedPackedFunc<R(Args...)>;
  /*! \brief default constructor */
  TypedPackedFunc() {}
  /*! \brief constructor from null */
  TypedPackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief construct by wrap a PackedFunc
   *
   * Example usage:
   * \code
   * PackedFunc packed([](DECORDArgs args, DECORDRetValue *rv) {
   *   int x = args[0];
   *   *rv = x + 1;
   *  });
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(packed);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param packed The packed function
   */
  inline TypedPackedFunc(PackedFunc packed);  // NOLINT(*)
  /*!
   * \brief constructor from DECORDRetValue
   * \param value The DECORDRetValue
   */
  inline TypedPackedFunc(const DECORDRetValue& value);  // NOLINT(*)
  /*!
   * \brief constructor from DECORDArgValue
   * \param value The DECORDArgValue
   */
  inline TypedPackedFunc(const DECORDArgValue& value);  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TypedPackedFunc(const FLambda& typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(PackedFunc packed) {
    packed_ = packed;
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  inline R operator()(Args ...args) const;
  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator PackedFunc() const {
    return packed();
  }
  /*!
   * \return reference the internal PackedFunc
   */
  const PackedFunc& packed() const {
    return packed_;
  }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return packed_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return packed_ != nullptr;
  }

 private:
  friend class DECORDRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template<typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

/*! \brief Arguments into DECORD functions. */
class DECORDArgs {
 public:
  const DECORDValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  DECORDArgs(const DECORDValue* values,
          const int* type_codes,
          int num_args)
      : values(values),
        type_codes(type_codes),
        num_args(num_args) { }
  /*! \return size of the arguments */
  inline int size() const;
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline DECORDArgValue operator[](int i) const;
};

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* TypeCode2Str(int type_code);

/*!
 * \brief convert a string to DECORD type.
 * \param s The string to be converted.
 * \return The corresponding decord type.
 */
inline DECORDType String2DECORDType(std::string s);

/*!
 * \brief convert a DECORD type to string.
 * \param t The type to be converted.
 * \return The corresponding decord type in string.
 */
inline std::string DECORDType2String(DECORDType t);

// macro to check type code.
#define DECORD_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

/*!
 * \brief Type traits to mark if a class is decord extension type.
 *
 * To enable extension type in C++ must be register () ed via marco.
 * DECORD_REGISTER_EXT_TYPE(TypeName) after defining this with this traits.
 *
 * Extension class can be passed and returned via PackedFunc in all decord runtime.
 * Internally extension class is stored as T*.
 *
 * \tparam T the typename
 */
template<typename T>
struct extension_class_info {
  static const int code = 0;
};

/*!
 * \brief Runtime function table about extension type.
 */
class ExtTypeVTable {
 public:
  /*! \brief function to be called to delete a handle */
  void (*destroy)(void* handle);
  /*! \brief function to be called when clone a handle */
  void* (*clone)(void* handle);
  /*!
   * \brief Register type
   * \tparam T The type to be register.
   * \return The registered vtable.
   */
  template <typename T>
  static inline ExtTypeVTable* Register_();
  /*!
   * \brief Get a vtable based on type code.
   * \param type_code The type code
   * \return The registered vtable.
   */
  DECORD_DLL static ExtTypeVTable* Get(int type_code);

 private:
  // Internal registration function.
  DECORD_DLL static ExtTypeVTable* RegisterInternal(int type_code, const ExtTypeVTable& vt);
};

/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class DECORDPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    DECORD_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64,
             std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    if (type_code_ == kArrayHandle) return value_.v_handle;
    DECORD_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator DLTensor*() const {
    if (type_code_ == kArrayHandle ||
        type_code_ == kNDArrayContainer) {
      return static_cast<DLTensor*>(value_.v_handle);
    } else {
      if (type_code_ == kNull) return nullptr;
      LOG(FATAL) << "Expected "
                 << "DLTensor* or NDArray but get "
                 << TypeCode2Str(type_code_);
      return nullptr;
    }
  }
  operator NDArray() const {
    if (type_code_ == kNull) return NDArray();
    DECORD_CHECK_TYPE_CODE(type_code_, kNDArrayContainer);
    return NDArray(static_cast<NDArray::Container*>(value_.v_handle));
  }
  operator DECORDContext() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kDECORDContext);
    return value_.v_ctx;
  }
  template<typename TExtension>
  const TExtension& AsExtension() const {
    CHECK_LT(type_code_, kExtEnd);
    return static_cast<TExtension*>(value_.v_handle)[0];
  }
  int type_code() const {
    return type_code_;
  }
  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template<typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

 protected:
  friend class DECORDArgsSetter;
  friend class DECORDRetValue;
  DECORDPODValue_() : type_code_(kNull) {}
  DECORDPODValue_(DECORDValue value, int type_code)
      : value_(value), type_code_(type_code) {}

  /*! \brief The value */
  DECORDValue value_;
  /*! \brief the type code */
  int type_code_;
};

/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and DECORDValue
 *
 *  Provides utilities to do type cast into other types.
 */
class DECORDArgValue : public DECORDPODValue_ {
 public:
  /*! \brief default constructor */
  DECORDArgValue() {}
  /*!
   * \brief constructor
   * \param value of the function
   * \param type_code The type code.
   */
  DECORDArgValue(DECORDValue value, int type_code)
      : DECORDPODValue_(value, type_code) {
  }
  // reuse converter from parent
  using DECORDPODValue_::operator double;
  using DECORDPODValue_::operator int64_t;
  using DECORDPODValue_::operator uint64_t;
  using DECORDPODValue_::operator int;
  using DECORDPODValue_::operator bool;
  using DECORDPODValue_::operator void*;
  using DECORDPODValue_::operator DLTensor*;
  using DECORDPODValue_::operator NDArray;
  using DECORDPODValue_::operator DECORDContext;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kDECORDType) {
      return DECORDType2String(operator DECORDType());
    } else if (type_code_ == kBytes) {
      DECORDByteArray* arr = static_cast<DECORDByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else {
      DECORD_CHECK_TYPE_CODE(type_code_, kStr);
      return std::string(value_.v_str);
    }
  }
  operator DECORDType() const {
    if (type_code_ == kStr) {
      return String2DECORDType(operator std::string());
    }
    // None type
    if (type_code_ == kNull) {
      DECORDType t;
      t.code = kHandle; t.bits = 0; t.lanes = 0;
      return t;
    }
    DECORD_CHECK_TYPE_CODE(type_code_, kDECORDType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    DECORD_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  const DECORDValue& value() const {
    return value_;
  }
  // Deferred extension handler.
  template<typename TNodeRef>
  inline TNodeRef AsNodeRef() const;
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;
  template<typename TNodeRef,
           typename = typename std::enable_if<
             std::is_class<TNodeRef>::value>::type>
  inline bool IsNodeType() const;
  inline operator HalideIR::Type() const;
  inline operator HalideIR::Expr() const;
  inline operator decord::Integer() const;
  // get internal node ptr, if it is node
  inline NodePtr<Node>& node_sptr();
};

/*!
 * \brief Return Value container,
 *  Unlike DECORDArgValue, which only holds reference and do not delete
 *  the underlying container during destruction.
 *
 *  DECORDRetValue holds value and will manage the underlying containers
 *  when it stores a complicated data type.
 */
class DECORDRetValue : public DECORDPODValue_ {
 public:
  /*! \brief default constructor */
  DECORDRetValue() {}
  /*!
   * \brief move constructor from anoter return value.
   * \param other The other return value.
   */
  DECORDRetValue(DECORDRetValue&& other)
      : DECORDPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kNull;
  }
  /*! \brief destructor */
  ~DECORDRetValue() {
    this->Clear();
  }
  // reuse converter from parent
  using DECORDPODValue_::operator double;
  using DECORDPODValue_::operator int64_t;
  using DECORDPODValue_::operator uint64_t;
  using DECORDPODValue_::operator int;
  using DECORDPODValue_::operator bool;
  using DECORDPODValue_::operator void*;
  using DECORDPODValue_::operator DLTensor*;
  using DECORDPODValue_::operator DECORDContext;
  using DECORDPODValue_::operator NDArray;
  DECORDRetValue(const DECORDRetValue& other) : DECORDPODValue_() {
    this->Assign(other);
  }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kDECORDType) {
      return DECORDType2String(operator DECORDType());
    } else if (type_code_ == kBytes) {
      return *ptr<std::string>();
    }
    DECORD_CHECK_TYPE_CODE(type_code_, kStr);
    return *ptr<std::string>();
  }
  operator DECORDType() const {
    if (type_code_ == kStr) {
      return String2DECORDType(operator std::string());
    }
    DECORD_CHECK_TYPE_CODE(type_code_, kDECORDType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    DECORD_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    DECORD_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  // Assign operators
  DECORDRetValue& operator=(DECORDRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kNull;
    return *this;
  }
  DECORDRetValue& operator=(double value) {
    this->SwitchToPOD(kDLFloat);
    value_.v_float64 = value;
    return *this;
  }
  DECORDRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kNull);
    value_.v_handle = value;
    return *this;
  }
  DECORDRetValue& operator=(void* value) {
    this->SwitchToPOD(kHandle);
    value_.v_handle = value;
    return *this;
  }
  DECORDRetValue& operator=(int64_t value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  DECORDRetValue& operator=(int value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  DECORDRetValue& operator=(DECORDContext value) {
    this->SwitchToPOD(kDECORDContext);
    value_.v_ctx = value;
    return *this;
  }
  DECORDRetValue& operator=(DECORDType t) {
    this->SwitchToPOD(kDECORDType);
    value_.v_type = t;
    return *this;
  }
  DECORDRetValue& operator=(bool value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  DECORDRetValue& operator=(std::string value) {
    this->SwitchToClass(kStr, value);
    return *this;
  }
  DECORDRetValue& operator=(DECORDByteArray value) {
    this->SwitchToClass(kBytes, std::string(value.data, value.size));
    return *this;
  }
  DECORDRetValue& operator=(NDArray other) {
    this->Clear();
    type_code_ = kNDArrayContainer;
    value_.v_handle = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  DECORDRetValue& operator=(PackedFunc f) {
    this->SwitchToClass(kFuncHandle, f);
    return *this;
  }
  template<typename FType>
  DECORDRetValue& operator=(const TypedPackedFunc<FType>& f) {
    return operator=(f.packed());
  }
  DECORDRetValue& operator=(Module m) {
    this->SwitchToClass(kModuleHandle, m);
    return *this;
  }
  DECORDRetValue& operator=(const DECORDRetValue& other) {  // NOLINT(*0
    this->Assign(other);
    return *this;
  }
  DECORDRetValue& operator=(const DECORDArgValue& other) {
    this->Assign(other);
    return *this;
  }
  template<typename T,
           typename = typename std::enable_if<
             extension_class_info<T>::code != 0>::type>
  DECORDRetValue& operator=(const T& other) {
    this->SwitchToClass<T>(
        extension_class_info<T>::code, other);
    return *this;
  }
  /*!
   * \brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources is moved to front-end and
   *  the front end should take charge in managing them.
   *
   * \param ret_value The return value.
   * \param ret_type_code The return type code.
   */
  void MoveToCHost(DECORDValue* ret_value,
                   int* ret_type_code) {
    // cannot move str; need specially handle.
    CHECK(type_code_ != kStr && type_code_ != kBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kNull;
  }
  /*! \return The value field, if the data is POD */
  const DECORDValue& value() const {
    CHECK(type_code_ != kNodeHandle &&
          type_code_ != kFuncHandle &&
          type_code_ != kModuleHandle &&
          type_code_ != kStr) << "DECORDRetValue.value can only be used for POD data";
    return value_;
  }
  // NodeRef related extenstions: in decord/packed_func_ext.h
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;
  template<typename TNodeRef>
  inline TNodeRef AsNodeRef() const;
  inline DECORDRetValue& operator=(const NodeRef& other);
  inline DECORDRetValue& operator=(const NodePtr<Node>& other);
  // type related
  inline operator HalideIR::Type() const;
  inline DECORDRetValue& operator=(const HalideIR::Type& other);

 private:
  template<typename T>
  void Assign(const T& other) {
    switch (other.type_code()) {
      case kStr: {
        SwitchToClass<std::string>(kStr, other);
        break;
      }
      case kBytes: {
        SwitchToClass<std::string>(kBytes, other);
        break;
      }
      case kFuncHandle: {
        SwitchToClass<PackedFunc>(kFuncHandle, other);
        break;
      }
      case kModuleHandle: {
        SwitchToClass<Module>(kModuleHandle, other);
        break;
      }
      case kNDArrayContainer: {
        *this = other.operator NDArray();
        break;
      }
      case kNodeHandle: {
        SwitchToClass<NodePtr<Node> >(
            kNodeHandle, *other.template ptr<NodePtr<Node> >());
        break;
      }
      default: {
        if (other.type_code() < kExtBegin) {
          SwitchToPOD(other.type_code());
          value_ = other.value_;
        } else {
#if DECORD_RUNTIME_HEADER_ONLY
          LOG(FATAL) << "Header only mode do not support ext type";
#else
          this->Clear();
          type_code_ = other.type_code();
          value_.v_handle =
              (*(ExtTypeVTable::Get(other.type_code())->clone))(
                  other.value().v_handle);
#endif
        }
        break;
      }
    }
  }
  // get the internal container.
  void SwitchToPOD(int type_code) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
    }
  }
  template<typename T>
  void SwitchToClass(int type_code, T v) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
      value_.v_handle = new T(v);
    } else {
      *static_cast<T*>(value_.v_handle) = v;
    }
  }
  void Clear() {
    if (type_code_ == kNull) return;
    switch (type_code_) {
      case kStr: delete ptr<std::string>(); break;
      case kFuncHandle: delete ptr<PackedFunc>(); break;
      case kModuleHandle: delete ptr<Module>(); break;
      case kNodeHandle: delete ptr<NodePtr<Node> >(); break;
      case kNDArrayContainer: {
        static_cast<NDArray::Container*>(value_.v_handle)->DecRef();
        break;
      }
    }
    if (type_code_ > kExtBegin) {
#if DECORD_RUNTIME_HEADER_ONLY
          LOG(FATAL) << "Header only mode do not support ext type";
#else
      (*(ExtTypeVTable::Get(type_code_)->destroy))(value_.v_handle);
#endif
    }
    type_code_ = kNull;
  }
};

// implementation details
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kNodeHandle: return "NodeHandle";
    case kArrayHandle: return "ArrayHandle";
    case kDECORDType: return "DECORDType";
    case kDECORDContext: return "DECORDContext";
    case kFuncHandle: return "FunctionHandle";
    case kModuleHandle: return "ModuleHandle";
    case kNDArrayContainer: return "NDArrayContainer";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
inline std::ostream& operator<<(std::ostream& os, DECORDType t) {  // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool"; return os;
  }
  os << TypeCode2Str(t.code);
  if (t.code == kHandle) return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}

#endif

inline std::string DECORDType2String(DECORDType t) {
  if (t.bits == 0) return "";
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::ostringstream os;
  os << t;
  return os.str();
#else
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    return "bool";
  }
  repr += TypeCode2Str(t.code);
  if (t.code == kHandle) return repr;
  repr += std::to_string(static_cast<int>(t.bits));
  if (t.lanes != 1) {
    repr += "x" + std::to_string(static_cast<int>(t.lanes));
  }
  return repr;
#endif
}

inline DECORDType String2DECORDType(std::string s) {
  DECORDType t;
  // handle None type
  if (s.length() == 0) {
    t.bits = 0; t.lanes = 0; t.code = kHandle;
    return t;
  }
  t.bits = 32; t.lanes = 1;
  const char* scan;
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;  scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt; scan = s.c_str() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat; scan = s.c_str() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.c_str() + 6;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else {
    scan = s.c_str();
    LOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, &endpt, 10));
  }
  CHECK(endpt == s.c_str() + s.length()) << "unknown type " << s;
  return t;
}

inline DECORDArgValue DECORDArgs::operator[](int i) const {
  CHECK_LT(i, num_args)
      << "not enough argument passed, "
      << num_args << " passed"
      << " but request arg[" << i << "].";
  return DECORDArgValue(values[i], type_codes[i]);
}

inline int DECORDArgs::size() const {
  return num_args;
}

inline void PackedFunc::CallPacked(DECORDArgs args, DECORDRetValue* rv) const {
  body_(args, rv);
}

inline PackedFunc::FType PackedFunc::body() const {
  return body_;
}



// internal namespace
namespace detail {

template<bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template<typename T, typename ...Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I+1), F>
        ::run(f, std::forward<Args>(args)...);
  }
};

template<std::size_t I, typename F>
struct for_each_dispatcher<true, I, F>  {
  static void run(const F& f) {}  // NOLINT(*)
};

template<typename F, typename ...Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>
      ::run(f, std::forward<Args>(args)...);
}
}  // namespace detail

/* \brief argument settter to PackedFunc */
class DECORDArgsSetter {
 public:
  DECORDArgsSetter(DECORDValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template<typename T,
           typename = typename std::enable_if<
             std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    CHECK_LE(value,
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, const DECORDArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kArrayHandle;
  }
  void operator()(size_t i, DECORDContext value) const {
    values_[i].v_ctx = value;
    type_codes_[i] = kDECORDContext;
  }
  void operator()(size_t i, DECORDType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kDECORDType;
  }
  void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kStr;
  }
  // setters for container type
  // They must be reference(instead of const ref)
  // to make sure they are alive in the tuple(instead of getting converted)
  void operator()(size_t i, const std::string& value) const {  // NOLINT(*)
    values_[i].v_str = value.c_str();
    type_codes_[i] = kStr;
  }
  void operator()(size_t i, const DECORDByteArray& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<DECORDByteArray*>(&value);
    type_codes_[i] = kBytes;
  }
  void operator()(size_t i, const PackedFunc& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<PackedFunc*>(&value);
    type_codes_[i] = kFuncHandle;
  }
  template<typename FType>
  void operator()(size_t i, const TypedPackedFunc<FType>& value) const {  // NOLINT(*)
    operator()(i, value.packed());
  }
  void operator()(size_t i, const Module& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<Module*>(&value);
    type_codes_[i] = kModuleHandle;
  }
  void operator()(size_t i, const NDArray& value) const {  // NOLINT(*)
    values_[i].v_handle = value.data_;
    type_codes_[i] = kNDArrayContainer;
  }
  void operator()(size_t i, const DECORDRetValue& value) const {  // NOLINT(*)
    if (value.type_code() == kStr) {
      values_[i].v_str = value.ptr<std::string>()->c_str();
      type_codes_[i] = kStr;
    } else {
      CHECK_NE(value.type_code(), kBytes) << "not handled.";
      values_[i] = value.value_;
      type_codes_[i] = value.type_code();
    }
  }
  // extension
  template<typename T,
           typename = typename std::enable_if<
             extension_class_info<T>::code != 0>::type>
  inline void operator()(size_t i, const T& value) const;
  // NodeRef related extenstions: in decord/packed_func_ext.h
  inline void operator()(size_t i, const NodeRef& other) const;  // NOLINT(*)
  inline void operator()(size_t i, const HalideIR::Type& t) const;

 private:
  /*! \brief The values fields */
  DECORDValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template<typename... Args>
inline DECORDRetValue PackedFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  DECORDValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(DECORDArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  DECORDRetValue rv;
  body_(DECORDArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

namespace detail {
template<typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template<typename ...Args>
  static void run(const F& f,
                  const DECORDArgs& args_pack,
                  DECORDRetValue* rv,
                  Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>
        ::run(f, args_pack, rv,
              std::forward<Args>(unpacked_args)...,
              args_pack[index]);
  }
};

template<typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const DECORDArgs& args_pack,
                  DECORDRetValue* rv,
                  Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template<int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const DECORDArgs& args_pack,
                  DECORDRetValue* rv,
                  Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename R, int nargs, typename F>
inline void unpack_call(const F& f, const DECORDArgs& args, DECORDRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template<typename R, typename ...Args>
inline R call_packed(const PackedFunc& pf, Args&& ...args) {
  return R(pf(std::forward<Args>(args)...));
}

template<typename R>
struct typed_packed_call_dispatcher {
  template<typename ...Args>
  static inline R run(const PackedFunc& pf, Args&& ...args) {
    return pf(std::forward<Args>(args)...);
  }
};

template<>
struct typed_packed_call_dispatcher<void> {
  template<typename ...Args>
  static inline void run(const PackedFunc& pf, Args&& ...args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed)
  : packed_(packed) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const DECORDRetValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const DECORDArgValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
template<typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const DECORDArgs& args, DECORDRetValue* rv) {
      detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
    });
}

template<typename R, typename ...Args>
inline R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>
      ::run(packed_, std::forward<Args>(args)...);
}

// extension and node type handling
namespace detail {
template<typename T, typename TSrc, bool is_ext>
struct DECORDValueCast {
  static T Apply(const TSrc* self) {
    return self->template AsNodeRef<T>();
  }
};

template<typename T, typename TSrc>
struct DECORDValueCast<T, TSrc, true> {
  static T Apply(const TSrc* self) {
    return self->template AsExtension<T>();
  }
};
}  // namespace detail

template<typename T, typename>
inline DECORDArgValue::operator T() const {
  return detail::
      DECORDValueCast<T, DECORDArgValue, extension_class_info<T>::code != 0>
      ::Apply(this);
}

template<typename T, typename>
inline DECORDRetValue::operator T() const {
  return detail::
      DECORDValueCast<T, DECORDRetValue, extension_class_info<T>::code != 0>
      ::Apply(this);
}

template<typename T, typename>
inline void DECORDArgsSetter::operator()(size_t i, const T& value) const {
  static_assert(extension_class_info<T>::code != 0,
                "Need to have extesion code");
  type_codes_[i] = extension_class_info<T>::code;
  values_[i].v_handle = const_cast<T*>(&value);
}

// extension type handling
template<typename T>
struct ExtTypeInfo {
  static void destroy(void* handle) {
    delete static_cast<T*>(handle);
  }
  static void* clone(void* handle) {
    return new T(*static_cast<T*>(handle));
  }
};

template<typename T>
inline ExtTypeVTable* ExtTypeVTable::Register_() {
  const int code = extension_class_info<T>::code;
  static_assert(code != 0,
                "require extension_class_info traits to be declared with non-zero code");
  ExtTypeVTable vt;
  vt.clone = ExtTypeInfo<T>::clone;
  vt.destroy = ExtTypeInfo<T>::destroy;
  return ExtTypeVTable::RegisterInternal(code, vt);
}

// Implement Module::GetFunction
// Put implementation in this file so we have seen the PackedFunc
inline PackedFunc Module::GetFunction(const std::string& name, bool query_imports) {
  PackedFunc pf = node_->GetFunction(name, node_);
  if (pf != nullptr) return pf;
  if (query_imports) {
    for (const Module& m : node_->imports_) {
      pf = m.node_->GetFunction(name, m.node_);
      if (pf != nullptr) return pf;
    }
  }
  return pf;
}
}  // namespace runtime
}  // namespace decord
#endif  // DECORD_RUNTIME_PACKED_FUNC_H_
