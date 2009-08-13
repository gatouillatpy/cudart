
#ifndef _CUDA_LONG1
#define _CUDA_LONG1

#include "_cuda_long.h"

static __inline__ __host__ __device__ long1 make_long1( float1 a )
{
	return make_long1( long(a.x) );
}

static __inline__ __host__ __device__ long1 min( long1 a, long1 b )
{
	long1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ long1 max( long1 a, long1 b )
{
	long1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ long1 operator-( long1 a )
{
	return make_long1( -a.x );
}

static __inline__ __host__ __device__ long1 operator+( long1 a, long1 b )
{
	return make_long1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( long1& a, long1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ long1 operator-( long1 a, long1 b )
{
	return make_long1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( long1& a, long1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ long1 operator*( long1 a, long1 b )
{
	return make_long1( a.x * b.x );
}

static __inline__ __host__ __device__ long1 operator*( long1 a, long s )
{
	return make_long1( a.x * s );
}

static __inline__ __host__ __device__ long1 operator*( long s, long1 a )
{
	return make_long1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( long1& a, long s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ long1 operator/( long1 a, long1 b )
{
	return make_long1( a.x / b.x );
}

static __inline__ __host__ __device__ long1 operator/( long1 a, long s )
{
	return make_long1( a.x / s );
}

static __inline__ __host__ __device__ long1 operator/( long s, long1 a )
{
	return make_long1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( long1& a, long s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ long1 clamp( long1 v, long a, long b )
{
	long1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ long1 clamp( long1 v, long1 a, long1 b )
{
	long1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ long1 abs( long1 v )
{
	return make_long1( abs( v.x ) );
}

#endif
