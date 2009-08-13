
#ifndef _CUDA_UINT1
#define _CUDA_UINT1

#include "_cuda_uint.h"

static __inline__ __host__ __device__ uint1 make_uint1( float1 a )
{
	return make_uint1( uint(a.x) );
}

static __inline__ __host__ __device__ uint1 min( uint1 a, uint1 b )
{
	uint1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ uint1 max( uint1 a, uint1 b )
{
	uint1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ uint1 operator+( uint1 a, uint1 b )
{
	return make_uint1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( uint1& a, uint1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ uint1 operator-( uint1 a, uint1 b )
{
	return make_uint1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( uint1& a, uint1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ uint1 operator*( uint1 a, uint1 b )
{
	return make_uint1( a.x * b.x );
}

static __inline__ __host__ __device__ uint1 operator*( uint1 a, uint s )
{
	return make_uint1( a.x * s );
}

static __inline__ __host__ __device__ uint1 operator*( uint s, uint1 a )
{
	return make_uint1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( uint1& a, uint s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ uint1 operator/( uint1 a, uint1 b )
{
	return make_uint1( a.x / b.x );
}

static __inline__ __host__ __device__ uint1 operator/( uint1 a, uint s )
{
	return make_uint1( a.x / s );
}

static __inline__ __host__ __device__ uint1 operator/( uint s, uint1 a )
{
	return make_uint1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( uint1& a, uint s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ uint1 clamp( uint1 v, uint a, uint b )
{
	uint1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ uint1 clamp( uint1 v, uint1 a, uint1 b )
{
	uint1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

#endif
