#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif  //__cplusplus


void gpu_kernel(int size, float alpha, float *d_x, float *d_y);

#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  // KERNEL_H_
