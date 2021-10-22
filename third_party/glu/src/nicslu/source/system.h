#ifndef __SYSTEM__
#define __SYSTEM__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//return the number of cores. if hyper-thread is supported and enabled, it returns 2X number
int GetProcessorNumber();

//linux does not support peakmem and peakvmem
int GetProcessMemory(size_t *mem, size_t *peakmem, size_t *vmem, size_t *peakvmem);

//32 or 64
int GetBitWidth();

//virtual memory means the maximum virtual address space of a process
int GetSystemMemory(unsigned long long *phys, unsigned long long *availphys, unsigned long long *virt);

#ifdef __cplusplus
}
#endif

#endif
