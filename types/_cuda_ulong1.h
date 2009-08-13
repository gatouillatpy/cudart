
#ifndef _CUDA_ULONG1
#define _CUDA_ULONG1

#include "_cuda_ulong.h"

static __inline__ __host__ __device__ ulong1 make_ulong1( float1 a )
{
	return make_ulong1( ulong(a.x) );
}

static __inline__ __host__ __device__ ulong1 min( ulong1 a, ulong1 b )
{
	ulong1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ ulong1 max( ulong1 a, ulong1 b )
{
	ulong1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ ulong1 operator+( ulong1 a, ulong1 b )
{
	return make_ulong1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( ulong1& a, ulong1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ ulong1 operator-( ulong1 a, ulong1 b )
{
	return make_ulong1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( ulong1& a, ulong1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ ulong1 operator*( ulong1 a, ulong1 b )
{
	return make_ulong1( a.x * b.x );
}

static __inline__ __host__ __device__ ulong1 operator*( ulong1 a, ulong s )
{
	return make_ulong1( a.x * s );
}

static __inline__ __host__ __device__ ulong1 operator*( ulong s, ulong1 a )
{
	return make_ulong1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( ulong1& a, ulong s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ ulong1 operator/( ulong1 a, ulong1 b )
{
	return make_ulong1( a.x / b.x );
}

static __inline__ __host__ __device__ ulong1 operator/( ulong1 a, ulong s )
{
	return make_ulong1( a.x / s );
}

static __inline__ __host__ __device__ ulong1 operator/( ulong s, ulong1 a )
{
	return make_ulong1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( ulong1& a, ulong s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ ulong1 clamp( ulong1 v, ulong a, ulong b )
{
	ulong1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ ulong1 clamp( ulong1 v, ulong1 a, ulong1 b )
{
	ulong1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

#endif
