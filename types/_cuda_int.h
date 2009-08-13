
#ifndef _CUDA_INT
#define _CUDA_INT

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ int min( int a, int b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ int max( int a, int b )
	{
		return a > b ? a : b;
	}

#else

	static __inline__ __device__ int __float_as_ordered_int( float f )
	{
		int i = __float_as_int( f );

		return ( i >= 0 ) ? i : i ^ 0x7FFFFFFF;
	}

#endif

static __inline__ __device__ __host__ int clamp( int f, int a, int b )
{
	return max( a, min( f, b ) );
}

#endif
