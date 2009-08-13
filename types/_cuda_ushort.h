
#ifndef _CUDA_USHORT
#define _CUDA_USHORT

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ ushort min( ushort a, ushort b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ ushort max( ushort a, ushort b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ ushort clamp( ushort f, ushort a, ushort b )
{
	return max( a, min( f, b ) );
}

#endif
