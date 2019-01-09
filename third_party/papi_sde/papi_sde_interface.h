#ifndef PAPI_SDE_INTERFACE_H
#define PAPI_SDE_INTERFACE_H

#include <stdint.h>
#include <stdlib.h>

#define PAPI_SDE_RO 0x00
#define PAPI_SDE_RW 0x01
#define PAPI_SDE_DELTA 0x00
#define PAPI_SDE_INSTANT 0x10

#define PAPI_SDE_long_long 0x0
#define PAPI_SDE_int 0x1
#define PAPI_SDE_double 0x2
#define PAPI_SDE_float 0x3

#define PAPI_SDE_SUM 0x0
#define PAPI_SDE_MAX 0x1
#define PAPI_SDE_MIN 0x2


#define GET_FLOAT_SDE(x) *((float *)&x)
#define GET_DOUBLE_SDE(x) *((double *)&x)
/*
 * GET_SDE_RECORDER_ADDRESS() USAGE EXAMPLE:
 * If SDE recorder logs values of type 'double':
 *     double *ptr = GET_SDE_RECORDER_ADDRESS(papi_event_value[6], double);
 *     for (j=0; j<CNT; j++)
 *        printf("    %d: %.4e\n",j, ptr[j]);
 */
#define GET_SDE_RECORDER_ADDRESS(x, rcrd_type) ((rcrd_type *)x)


typedef long long int (*papi_sde_fptr_t)(void *);
typedef int (*papi_sde_cmpr_fptr_t)(void *);
typedef void *papi_handle_t;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct papi_sde_fptr_struct_s {
    papi_handle_t (*init)(const char *lib_name);
    int (*register_counter)(void *handle, const char *event_name, int mode,
                            int type, void *counter);
    int (*register_fp_counter)(void *handle, const char *event_name, int mode,
                               int type, papi_sde_fptr_t fp_counter,
                               void *param);
    int (*unregister_counter)(void *handle, const char *event_name);
    int (*describe_counter)(void *handle, const char *event_name,
                            const char *event_description);
    int (*add_counter_to_group)(void *handle, const char *event_name,
                                const char *group_name, uint32_t group_flags);
    int (*create_counter)(papi_handle_t handle, const char *event_name,
                          int cntr_type, void **cntr_handle);
    int (*inc_counter)(papi_handle_t cntr_handle, long long int increment);
    int (*create_recorder)(papi_handle_t handle, const char *event_name,
                           size_t typesize,
                           int (*cmpr_func_ptr)(const void *p1, const void *p2),
                           void **record_handle);
    int (*record)(void *record_handle, size_t typesize, void *value);
    int (*reset_recorder)(void *record_handle);
    int (*reset_counter)(void *cntr_handle);
} papi_sde_fptr_struct_t;

papi_handle_t papi_sde_init(const char *name_of_library);
int papi_sde_register_counter(papi_handle_t handle, const char *event_name,
                              int cntr_mode, int cntr_type, void *counter);
int papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name,
                                 int cntr_mode, int cntr_type,
                                 papi_sde_fptr_t func_ptr, void *param);
int papi_sde_unregister_counter(void *handle, const char *event_name);
int papi_sde_describe_counter(papi_handle_t handle, const char *event_name,
                              const char *event_description);
int papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name,
                                  const char *group_name, uint32_t group_flags);
int papi_sde_create_counter(papi_handle_t handle, const char *event_name,
                            int cntr_type, void **cntr_handle);
int papi_sde_inc_counter(void *cntr_handle, long long int increment);
int papi_sde_create_recorder(
    papi_handle_t handle, const char *event_name, size_t typesize,
    int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle);
int papi_sde_record(void *record_handle, size_t typesize, void *value);
int papi_sde_reset_recorder(void *record_handle);
int papi_sde_reset_counter(void *cntr_handle);
void *papi_sde_get_counter_handle(papi_handle_t handle, const char *event_name);

int papi_sde_compare_long_long(const void *p1, const void *p2);
int papi_sde_compare_int(const void *p1, const void *p2);
int papi_sde_compare_double(const void *p1, const void *p2);
int papi_sde_compare_float(const void *p1, const void *p2);

papi_handle_t papi_sde_hook_list_events(papi_sde_fptr_struct_t *fptr_struct);
#ifdef __cplusplus
}
#endif

#define POPULATE_SDE_FPTR_STRUCT(_A_)                             \
    do {                                                          \
        _A_.init = papi_sde_init;                                 \
        _A_.register_counter = papi_sde_register_counter;         \
        _A_.register_fp_counter = papi_sde_register_fp_counter;   \
        _A_.unregister_counter = papi_sde_unregister_counter;     \
        _A_.describe_counter = papi_sde_describe_counter;         \
        _A_.add_counter_to_group = papi_sde_add_counter_to_group; \
        _A_.create_counter = papi_sde_create_counter;             \
        _A_.inc_counter = papi_sde_inc_counter;                   \
        _A_.create_recorder = papi_sde_create_recorder;           \
        _A_.record = papi_sde_record;                             \
        _A_.reset_recorder = papi_sde_reset_recorder;             \
        _A_.reset_counter = papi_sde_reset_counter;               \
    } while (0)

#endif
