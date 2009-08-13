
#ifndef _CUDA_ULONG
#define _CUDA_ULONG

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

#endif

static __inline__ __host__ __device__ ulong min( ulong a, ulong b )
{
	return a < b ? a : b;
}

static __inline__ __host__ __device__ ulong max( ulong a, ulong b )
{
	return a > b ? a : b;
}

static __inline__ __device__ __host__ ulong clamp( ulong f, ulong a, ulong b )
{
	return max( a, min( f, b ) );
}

#endif
