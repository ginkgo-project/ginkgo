/*C version timer*/
/*this program is used to get the absolute runtime of a piece of code. it is the real time, not cpu time*/
/*on Linux, link with -lrt*/
/*last modified: jun 9, 2013*/
/*author: Chen, Xiaoming*/

#ifndef __TIMER_C__
#define __TIMER_C__

#include <time.h>

#ifdef _WIN32

/*for windows*/
typedef struct tagSTimer
{
	__int64 freq;
	__int64 start;
	__int64 stop;
} STimer;

#else

/*for linux*/
#include <sys/time.h>
typedef struct tagSTimer
{
	struct timespec start;
	struct timespec stop;
} STimer;

#endif

/*interfaces*/
#ifdef __cplusplus
extern "C" {
#endif

/*length of fmt must > 20. if ok, return fmt*/
char	*TimerGetLocalTime(char *fmt);

/*if ok , return 0; otherwise return -1*/
int		TimerInit(STimer *tmr);

/*if ok , return 0; otherwise return -1*/
int		TimerStart(STimer *tmr);

/*if ok , return 0; otherwise return -1*/
int		TimerStop(STimer *tmr);

/*if ok , return stop-start; otherwise return -1.0. runtime is in seconds*/
double	TimerGetRuntime(STimer *tmr);

#ifdef __cplusplus
}
#endif

#endif
