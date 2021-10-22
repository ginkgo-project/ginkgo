/**************************************************/
//math.h
//to replace <math.h>
/**************************************************/

#ifndef __MATH__
#define __MATH__

#include <math.h>
#include <float.h>

#define isNaN(x)		((x) != (x))

#ifdef _WIN32
#define ISNAN(x)	_isnan(x)
#define ISINF(x)	(!_finite(x) && !_isnan(x))
#define FINITE(x)	_finite(x)
#else
#define ISNAN(x)	isnan(x)
#define ISINF(x)	isinf(x)
#define FINITE(x)	finite(x)
#endif

#ifndef MAX
#define MAX(a, b)					(((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b)					(((a) < (b)) ? (a) : (b))
#endif
#ifndef ABS
#define ABS(x)						((x) > 0 ? (x) : (-(x)))
#endif
#ifndef SIGNOF
#define SIGNOF(x)					((x) < 0 ? -1 : 1)
#endif

#define CONST_PI					(3.1415926535897932384626)
#define CONST_E						(2.7182818284590452353603)

#endif
