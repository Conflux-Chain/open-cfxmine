#pragma once

#include "params.cuh"
#include "structs.cuh"

// DAG
__constant__ u32 d_dag_size;
__constant__ hash256_t *d_dag;
__constant__ u32 d_light_size;
__constant__ hash64_t *d_light;
// changed across headers
__constant__ hash32_t d_header;
__constant__ hash32_t d_boundary;
__constant__ u32 d_x[OCTOPUS_N];
