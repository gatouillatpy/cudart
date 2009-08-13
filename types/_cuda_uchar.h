
#ifndef _CUDA_UCHAR
#define _CUDA_UCHAR

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ uchar min( uchar a, uchar b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ uchar max( uchar a, uchar b )
	{
		return a > b ? a : b;
	}

#endif

static __inline__ __device__ __host__ uchar clamp( uchar f, uchar a, uchar b )
{
	return max( a, min( f, b ) );
}

#endif
