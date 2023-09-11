#include "cuda/base/types.hpp"


#if defined(__CUDACC__)

#define BFLOAT_FRIEND_OPERATOR(_op, _opeq)                                  \
    __device__ __forceinline__ __nv_bfloat16 operator _op(                  \
        const __nv_bfloat16& lhs, const __nv_bfloat16& rhs)                 \
    {                                                                       \
        return static_cast<__nv_bfloat16>(static_cast<float>(lhs)           \
                                              _op static_cast<float>(rhs)); \
    }                                                                       \
    __device__ __forceinline__ __nv_bfloat16& operator _opeq(               \
        __nv_bfloat16& lhs, const __nv_bfloat16& rhs)                       \
    {                                                                       \
        lhs = static_cast<float>(lhs) _op static_cast<float>(rhs);          \
        return lhs;                                                         \
    }
BFLOAT_FRIEND_OPERATOR(+, +=)
BFLOAT_FRIEND_OPERATOR(-, -=)
BFLOAT_FRIEND_OPERATOR(*, *=)
BFLOAT_FRIEND_OPERATOR(/, /=)

__device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16& h)
{
    return h;
}
__device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16& h)
{
    return -float{h};
}

#endif