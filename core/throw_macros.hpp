#ifndef MSPARSE_CORE_THROW_MACROS_H_
#define MSPARSE_CORE_THROW_MACROS_H_

#include "error.h"


#include <typeinfo>


#define NOT_IMPLEMENTED \
{ throw ::msparse::NotImplemented(__FILE__, __LINE__, __func__); }


#define NOT_SUPPORTED(_obj) \
    ::msparse::NotSupported(__FILE__, __LINE__, __func__, typeid(_obj).name())


#define ASSERT_CONFORMANT(_operator, _vector) \
    if ((_operator)->get_num_cols() != (_vector)->get_num_rows() { \
        throw ::msparse::DimensionMismatch(__FILE__, __LINE__, __func__, \
                (_operator)->get_num_rows(), (_operator)->get_num_cols(), \
                (_vector)->get_num_rows(), (_vector)->get_num_cols()); \
    }


#define ASSERT_ALLOCATED(_ptr, _dev, _size) \
    if ((_ptr) == nullptr) { \
        throw ::msparse::AllocationError(__FILE__, __LINE__, _dev, _size); \
    }


#endif // MSPARSE_CORE_THROW_MACROS_H_

