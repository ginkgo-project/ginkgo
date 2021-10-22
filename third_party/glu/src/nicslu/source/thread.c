#include "thread.h"
#include <stdlib.h>

#ifdef _WIN32
/*windows*/
#include <limits.h>

int _CreateThread(thread_proc__t proc, void *arg, thread_id__t *id)
{
	if (proc == NULL || id == NULL) return -1;
	*id = (thread_id__t)_beginthreadex(NULL, 0, proc, arg, 0, NULL);
	return (*id==NULL ? -1 : 0);
}

int _WaitThreadExit(thread_id__t id)
{
	unsigned int rt = WaitForSingleObject(id, INFINITE);
	return (rt==WAIT_FAILED ? -1 : 0);
}

thread_id__t _GetCurrentThread()
{
	return GetCurrentThread();
}

int _BindThreadToCores(thread_id__t id, unsigned int *cores, int ct)
{
#ifdef NO_EXTENSION
	return -1;
#else
	DWORD_PTR set, old;
	int i;
	unsigned int len, c;
	int have;

	if (cores == NULL || ct <= 0) return -1;

	len = (unsigned int)(sizeof(DWORD_PTR));
	len <<= 3;

	set = (DWORD_PTR)0;
	have = 0;
	for (i=0; i<ct; ++i)
	{
		c = cores[i];
		if (c+1 < len)
		{
			set |= (((DWORD_PTR)1) << c);
			have = 1;
		}
	}

	if (have)
	{
		old = SetThreadAffinityMask(id, set);
		return (old==0 ? -1 : 0);
	}
	else return 0;
#endif
}

int _UnbindThreadFromCores(thread_id__t id)
{
#ifdef NO_EXTENSION
	return -1;
#else
	DWORD_PTR set, old;
	set = (DWORD_PTR)(-1);
	old = SetThreadAffinityMask(id, set);
	return (old==0 ? -1 : 0);
#endif
}

int _CreateLock(lock__t *lock)
{
	if (lock == NULL) return -1;
	*lock = CreateEvent(NULL, FALSE, TRUE, NULL);
	return (*lock==NULL ? -1 : 0);
}

int _Lock(lock__t *lock)
{
	unsigned int rt;
	if (lock == NULL) return -1;
	rt = WaitForSingleObject(*lock, INFINITE);
	return (rt==WAIT_FAILED ? -1 : 0);
}

int _Unlock(lock__t *lock)
{
	int rt;
	if (lock == NULL) return -1;
	rt = SetEvent(*lock);
	return (rt ? 0 : -1);
}

int _DestroyLock(lock__t *lock)
{
	int rt;
	if (lock == NULL) return -1;
	rt = CloseHandle(*lock);
	return (rt ? 0 : -1);
}

void _Delay(unsigned int ms)
{
	Sleep(ms);
}

int _CreateSemaphore(sem__t *sem, int initval)
{
	if (sem == NULL) return -1;
	*sem = CreateSemaphore(NULL, initval, INT_MAX, NULL);
	return (*sem==NULL ? -1 : 0);
}

int _WaitSemaphore(sem__t *sem)
{
	unsigned int rt;
	if (sem == NULL) return -1;
	rt = WaitForSingleObject(*sem, INFINITE);
	return (rt==WAIT_FAILED ? -1 : 0);
}

int _IncreaseSemaphore(sem__t *sem)
{
	int rt;
	if (sem == NULL) return -1;
	rt = ReleaseSemaphore(*sem, 1, NULL);
	return (rt ? 0 : -1);
}

int _DestroySemaphore(sem__t *sem)
{
	int rt;
	if (sem == NULL) return -1;
	rt = CloseHandle(*sem);
	return (rt ? 0 : -1);
}

int _CreateEvent(event__t *ev, int initval)
{
	if (ev == NULL) return -1;
	*ev = CreateEvent(NULL, TRUE, initval>0?TRUE:FALSE, NULL);
	return (*ev==NULL ? -1 : 0);
}

int _CreateEventA(event__t *ev, int initval)
{
	if (ev == NULL) return -1;
	*ev = CreateEvent(NULL, FALSE, initval>0?TRUE:FALSE, NULL);
	return (*ev==NULL ? -1 : 0);
}

int _WaitEvent(event__t *ev)
{
	unsigned int rt;
	if (ev == NULL) return -1;
	rt = WaitForSingleObject(*ev, INFINITE);
	return (rt==WAIT_FAILED ? -1 : 0);
}

int _WaitEventA(event__t *ev)
{
	unsigned int rt;
	if (ev == NULL) return -1;
	rt = WaitForSingleObject(*ev, INFINITE);
	return (rt==WAIT_FAILED ? -1 : 0);
}

int _SetEvent(event__t *ev)
{
	int rt;
	if (ev == NULL) return -1;
	rt = SetEvent(*ev);
	return (rt ? 0 : -1);
}

int _ResetEvent(event__t *ev)
{
	int rt;
	if (ev == NULL) return -1;
	rt = ResetEvent(*ev);
	return (rt ? 0 : -1);
}

int _DestroyEvent(event__t *ev)
{
	int rt;
	if (ev == NULL) return -1;
	rt = CloseHandle(*ev);
	return (rt ? 0 : -1);
}

int _InitializeCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	InitializeCriticalSection(cs);
	return 0;
}

int _EnterCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	EnterCriticalSection(cs);
	return 0;
}

int _LeaveCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	LeaveCriticalSection(cs);
	return 0;
}

int _DeleteCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	DeleteCriticalSection(cs);
	return 0;
}

#ifndef NO_ATOMIC
int _SpinInit(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	*spin = 0;
	return 0;
}

int _SpinLock(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	while (InterlockedExchange(spin, 1) != 0) {}
	return 0;
}

int _SpinUnlock(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	InterlockedExchange(spin, 0);
	return 0;
}
#endif

#else
/*linux*/
#include <sched.h>

int _CreateThread(thread_proc__t proc, void *arg, thread_id__t *id)
{
	return pthread_create(id, NULL, proc, arg);
}

int _WaitThreadExit(thread_id__t id)
{
	return pthread_join(id, NULL);
}

thread_id__t _GetCurrentThread()
{
	return pthread_self();
}

int _BindThreadToCores(thread_id__t id, unsigned int *cores, int ct)
{
#ifdef NO_EXTENSION
	return -1;
#else
	cpu_set_t set;
	int i;
	unsigned int c;
	int have;

	if (cores == NULL || ct <= 0) return -1;

	CPU_ZERO(&set);
	have = 0;
	for (i=0; i<ct; ++i)
	{
		c = cores[i];
		if (c+1 < CPU_SETSIZE)
		{
			CPU_SET(c, &set);
			have = 1;
		}
	}

	if (have)
	{
		return pthread_setaffinity_np(id, sizeof(cpu_set_t), &set);
	}
	else return 0;
#endif
}

int _UnbindThreadFromCores(thread_id__t id)
{
#ifdef NO_EXTENSION
	return -1;
#else
	int i;
	cpu_set_t set;
	CPU_ZERO(&set);
	for (i=0; i<CPU_SETSIZE; ++i)
	{
		CPU_SET(i, &set);
	}
	return pthread_setaffinity_np(id, sizeof(cpu_set_t), &set);
#endif
}

int _CreateLock(lock__t *lock)
{
	if (lock == NULL) return -1;
	return pthread_mutex_init(lock, NULL);
}

int _Lock(lock__t *lock)
{
	if (lock == NULL) return -1;
	return pthread_mutex_lock(lock);
}

int _Unlock(lock__t *lock)
{
	if (lock == NULL) return -1;
	return pthread_mutex_unlock(lock);
}

int _DestroyLock(lock__t *lock)
{
	if (lock == NULL) return -1;
	return pthread_mutex_destroy(lock);
}

void _Delay(unsigned int ms)
{
	usleep(ms*1000);
}

int _CreateSemaphore(sem__t *sem, int initval)
{
	if (sem == NULL) return -1;
	return sem_init(sem, 0, initval);
}

int _WaitSemaphore(sem__t *sem)
{
	if (sem == NULL) return -1;
	return sem_wait(sem);
}

int _IncreaseSemaphore(sem__t *sem)
{
	if (sem == NULL) return -1;
	return sem_post(sem);
}

int _DestroySemaphore(sem__t *sem)
{
	if (sem == NULL) return -1;
	return sem_destroy(sem);
}

int _CreateEvent(event__t *ev, int initval)
{
	if (ev == NULL) return -1;
	return sem_init(ev, 0, initval>0?1:0);
}

int _CreateEventA(event__t *ev, int initval)
{
	if (ev == NULL) return -1;
	return sem_init(ev, 0, initval>0?1:0);
}

int _WaitEvent(event__t *ev)
{
	if (ev == NULL) return -1;
	sem_wait(ev);
	return sem_post(ev);
}

int _WaitEventA(event__t *ev)
{
	if (ev == NULL) return -1;
	return sem_wait(ev);
}

int _SetEvent(event__t *ev)
{
	if (ev == NULL) return -1;
	return sem_post(ev);
}

int _ResetEvent(event__t *ev)
{
	if (ev == NULL) return -1;
	sem_trywait(ev);
	return 0;
}

int _DestroyEvent(event__t *ev)
{
	if (ev == NULL) return -1;
	return sem_destroy(ev);
}

int _InitializeCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	return pthread_mutex_init(cs, NULL);
}

int _EnterCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	return pthread_mutex_lock(cs);
}

int _LeaveCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	return pthread_mutex_unlock(cs);
}

int _DeleteCriticalSection(critical_section__t *cs)
{
	if (cs == NULL) return -1;
	return pthread_mutex_destroy(cs);
}

#ifndef NO_ATOMIC
int _SpinInit(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	*spin = 0;
	return 0;
}

int _SpinLock(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	while (__sync_fetch_and_or(spin, 1) != 0) {}
	return 0;
}

int _SpinUnlock(spin_lock__t *spin)
{
	if (spin == NULL) return -1;
	__sync_and_and_fetch(spin, 0);
	return 0;
}
#endif

#endif

void _SpinWaitInt(volatile int *a)
{
	while (0 == *a);
}

void _SpinWaitShort(volatile short *a)
{
	while (0 == *a);
}

void _SpinWaitChar(volatile char *a)
{
	while (0 == *a);
}

#ifdef _WIN32
void _SpinWaitInt64(volatile __int64 *a)
#else
void _SpinWaitInt64(volatile long long *a)
#endif
{
	while (0 == *a);
}

void _SpinWaitFloat(volatile float *a)
{
	while (0.f == *a);
}

void _SpinWaitDouble(volatile double *a)
{
	while (0. == *a);
}

void _SpinWaitSizeInt(volatile size_t *a)
{
	while (0 == *a);
}

void _SpinBarrier(int s, int n, volatile char *a)
{
	int i;
	volatile char *p;
	for (i=s; i<n; ++i)
	{
		p = &(a[i]);
		while (0 == *p);
	}
}

int _InitBarrier(barrier__t *barr, int threads)
{
	if (barr == NULL || threads <= 0) return -1;
	barr->cnt = threads;
	barr->reach = (int *)malloc(sizeof(int)*(threads+threads));
	if (barr->reach == NULL) return -1;
	barr->run = barr->reach + threads;
	memset(barr->reach, 0, sizeof(int)*(threads+threads));
	return 0;
}

int _Barrier(barrier__t *barr, int id, int end)
{
	if (barr == NULL || end <= 0 || id < 0 || id >= end) return -1;
	if (end > barr->cnt) return -1;

	if (id == 0)
	{
		int i;
		volatile int *reach, *run;
		for (i=1; i<end; ++i)
		{
			reach = &(barr->reach[i]);
			while (0 == *reach);
		}
		for (i=1; i<end; ++i)
		{
			reach = &(barr->reach[i]);
			run = &(barr->run[i]);
			*reach = 0;
			*run = 1;
		}
	}
	else
	{
		volatile int *run, *reach;
		run	= &(barr->run[id]);
		reach = &(barr->reach[id]);
		*reach = 1;
		while (0 == *run);
		*run = 0;
	}

	return 0;
}

int _DestroyBarrier(barrier__t *barr)
{
	if (barr == NULL) return -1;
	free(barr->reach);
	return 0;
}
