
#ifndef _CUDA_CHAR
#define _CUDA_CHAR

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ char min( char a, char b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ char max( char a, char b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ char clamp( char f, char a, char b )
{
	return max( a, min( f, b ) );
}

#endif
