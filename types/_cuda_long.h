
#ifndef _CUDA_LONG
#define _CUDA_LONG

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ long min( long a, long b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ long max( long a, long b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ long clamp( long f, long a, long b )
{
	return max( a, min( f, b ) );
}

#endif
