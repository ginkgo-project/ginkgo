/*common configurations for NICS software*/

#ifndef __NICS_CONFIG__
#define __NICS_CONFIG__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifdef _WIN32
#ifdef _M_X64
#define X64__
#endif
#else
#if (defined (__amd64__) || defined (__x86_64__))
#define X64__
#endif
#endif

#ifdef NICS_INT64
#ifdef X64__
#define INT64__
#else
#error NICS_INT64 IS ONLY USED FOR 64-BIT ARCHITECTURE!
#endif
#endif

/*return code*/
#define NICS_OK				(0)

/*constant*/
#ifndef TRUE
#define TRUE				(1)
#endif
#ifndef FALSE
#define FALSE				(0)
#endif

/*typedef*/
typedef unsigned char		byte__t;
typedef unsigned short		word__t;
typedef unsigned int		dword__t;
#ifdef _WIN32
typedef unsigned __int64	qword__t;
#else
typedef unsigned long long	qword__t;
#endif

typedef char				int8__t;
typedef unsigned char		uint8__t;
typedef short				int16__t;
typedef unsigned short		uint16__t;
typedef int					int32__t;
typedef unsigned int		uint32__t;
#ifdef _WIN32
typedef __int64				int64__t;
typedef unsigned __int64	uint64__t;
#else
typedef long long			int64__t;
typedef unsigned long long	uint64__t;
#endif

typedef char				char__t;
typedef unsigned char		bool__t;
#ifdef INT64__
typedef int64__t			int__t;
typedef uint64__t			uint__t;
#else
typedef int32__t			int__t;
typedef uint32__t			uint__t;
#endif
typedef double				real__t;
typedef size_t				size__t;

typedef struct __tag_complex
{
	real__t real;
	real__t image;
} complex__t;

#endif

/*length (in bytes) of all integer types in windows and linux, for mainstream compilers
				windows-32	windows-64	linux-32	linux-64
char			1			1			1			1
short			2			2			2			2
int				4			4			4			4
long			4			4			4			8
long long		8			8			8			8
size_t			4			8			4			8
*/
