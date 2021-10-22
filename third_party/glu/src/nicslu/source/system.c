#include "system.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <Windows.h>
#include <Psapi.h>
#else
#include <sys/types.h>
#include <unistd.h>
#include <sys/resource.h>
#endif

int GetProcessorNumber()
{
#ifdef _WIN32

	typedef void (WINAPI *PGNSI)(LPSYSTEM_INFO);
	SYSTEM_INFO si;
	PGNSI pfnGNSI = (PGNSI)GetProcAddress(GetModuleHandle("kernel32.dll"), "GetNativeSystemInfo");
	if (pfnGNSI != NULL)
	{
		pfnGNSI(&si);
	}
	else
	{
		GetSystemInfo(&si);
	}
	return si.dwNumberOfProcessors;

#else

/*	return sysconf(_SC_NPROCESSORS_CONF);*/
	return sysconf(_SC_NPROCESSORS_ONLN);

#endif
}

#ifndef _WIN32

#define PROC_NAME_LEN		64
#define THREAD_NAME_LEN		32
#define MAX_LINE			256

struct proc_info
{
	pid_t pid;
	pid_t tid;
	uid_t uid;
	gid_t gid;
	char state;
	unsigned long utime;
	unsigned long stime;
	unsigned long vss;
	unsigned long rss;
};

static int read_proc_stat(char *filename, struct proc_info *proc)
{
	FILE *file;
	char buf[MAX_LINE], *open_paren, *close_paren;

	file = fopen(filename, "r");
	if (file == NULL) return -1;
	fgets(buf, MAX_LINE, file);
	fclose(file);

	/* Split at first '(' and last ')' to get process name. */
	open_paren = strchr(buf, '(');
	close_paren = strrchr(buf, ')');
	if (open_paren == NULL || close_paren == NULL) return -1;

	*open_paren = *close_paren = '\0';
	sscanf(close_paren + 1, " %c %*d %*d %*d %*d %*d %*d %*d %*d %*d %*d "
		"%lu %lu %*d %*d %*d %*d %*d %*d %*d %lu %lu",
		&proc->state, &proc->utime, &proc->stime, &proc->vss, &proc->rss);

	return 0;
}

#endif

int GetProcessMemory(size_t *mem, size_t *peakmem, size_t *vmem, size_t *peakvmem)
{	
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS pmc;
	typedef BOOL (WINAPI *PGPMI)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
	PGPMI pfn;
	HMODULE dll;
#else
	pid_t pid;
	struct proc_info proc;
	char filename[4096];
#endif

	if (NULL == mem || NULL == peakmem || NULL == vmem || NULL == peakvmem) return -1;
	*mem = 0;
	*peakmem = 0;
	*vmem = 0;
	*peakvmem = 0;

#ifdef _WIN32
	
	dll = LoadLibrary("psapi.dll");
	if (NULL == dll) return -1;
	pfn = (PGPMI)GetProcAddress(dll, "GetProcessMemoryInfo");
	if (NULL == pfn)
	{
		FreeLibrary(dll);
		return -1;
	}

	memset(&pmc, 0, sizeof(PROCESS_MEMORY_COUNTERS));

	pfn(GetCurrentProcess(), &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	*mem = pmc.WorkingSetSize;
	*peakmem = pmc.PeakWorkingSetSize;
	*vmem = pmc.PagefileUsage;
	*peakvmem = pmc.PeakPagefileUsage;

	FreeLibrary(dll);
	return 0;

#else

	memset(&proc, 0, sizeof(struct proc_info));

	pid = getpid();
	proc.pid = pid;
	proc.tid = pid;

	sprintf(filename, "/proc/%d/stat", pid);
	read_proc_stat(filename, &proc);
	*mem = *peakmem = proc.rss * getpagesize();
	*vmem = *peakvmem = proc.vss;

	return 0;

#endif
}

int GetBitWidth()
{
	return (sizeof(void *) << 3);
}

int GetSystemMemory(unsigned long long *phys, unsigned long long *availphys, unsigned long long *virt)
{
#ifdef _WIN32
	MEMORYSTATUSEX ms;
#else
	long long page_size;
	struct rlimit rl;
#endif

	if (phys == NULL || virt == NULL || availphys == NULL) return -1;

#ifdef _WIN32
	ms.dwLength = sizeof(MEMORYSTATUSEX);
	GlobalMemoryStatusEx(&ms);
	*phys = ms.ullTotalPhys;
	*virt = ms.ullTotalVirtual;
	*availphys = ms.ullAvailPhys;
#else
	page_size = sysconf(_SC_PAGESIZE);
	*phys = sysconf(_SC_PHYS_PAGES) * page_size;
	*availphys = sysconf(_SC_AVPHYS_PAGES) * page_size;
	getrlimit(RLIMIT_AS, &rl);
	*virt = rl.rlim_cur;
#endif

	return 0;
}
