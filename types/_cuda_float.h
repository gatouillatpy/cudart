
#ifndef _CUDA_FLOAT
#define _CUDA_FLOAT

#include <cuda_runtime.h>

#ifndef __CUDACC__

	#include <math.h>

	static __inline__ __host__ __device__ float fminf( float a, float b )
	{
		return a < b ? a : b;
	}

	static __inline__ __host__ __device__ float fmaxf( float a, float b )
	{
		return a > b ? a : b;
	}

	static __inline__ __host__ __device__ float rsqrtf( float x )
	{
		return 1.0f / sqrtf(x);
	}

#else

	static __inline__ __device__ float __ordered_int_as_float( int i )
	{
		return __int_as_float( ( i >= 0 ) ? i : i ^ 0x7FFFFFFF );
	}

#endif

static __inline__ __device__ __host__ float lerp( float a, float b, float t )
{
	return a + t * ( b - a );
}

static __inline__ __device__ __host__ float clamp( float f, float a, float b )
{
	return fmaxf( a, fminf( f, b ) );
}

#endif
