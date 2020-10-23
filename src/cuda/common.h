#pragma once

#ifdef __CUDA_ARCH__
#define HOST __host__
#define HOST_DEVICE __host__ __device__ __forceinline__
#else
#define HOST
#define HOST_DEVICE
#endif