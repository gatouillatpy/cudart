
#ifndef _CUDA_SHORT
#define _CUDA_SHORT

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ short min( short a, short b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ short max( short a, short b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ short clamp( short f, short a, short b )
{
	return max( a, min( f, b ) );
}

#endif
