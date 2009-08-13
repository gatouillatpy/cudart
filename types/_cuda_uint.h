
#ifndef _CUDA_UINT
#define _CUDA_UINT

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ uint min( uint a, uint b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ uint max( uint a, uint b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ uint clamp( uint f, uint a, uint b )
{
	return max( a, min( f, b ) );
}

#endif
