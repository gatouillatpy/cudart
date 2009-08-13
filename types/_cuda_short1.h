
#ifndef _CUDA_SHORT1
#define _CUDA_SHORT1

#include "_cuda_short.h"

static __inline__ __host__ __device__ short1 make_short1( float1 a )
{
	return make_short1( short(a.x) );
}

static __inline__ __host__ __device__ short1 min( short1 a, short1 b )
{
	short1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ short1 max( short1 a, short1 b )
{
	short1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ short1 operator-( short1 a )
{
	return make_short1( -a.x );
}

static __inline__ __host__ __device__ short1 operator+( short1 a, short1 b )
{
	return make_short1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( short1& a, short1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ short1 operator-( short1 a, short1 b )
{
	return make_short1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( short1& a, short1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ short1 operator*( short1 a, short1 b )
{
	return make_short1( a.x * b.x );
}

static __inline__ __host__ __device__ short1 operator*( short1 a, short s )
{
	return make_short1( a.x * s );
}

static __inline__ __host__ __device__ short1 operator*( short s, short1 a )
{
	return make_short1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( short1& a, short s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ short1 operator/( short1 a, short1 b )
{
	return make_short1( a.x / b.x );
}

static __inline__ __host__ __device__ short1 operator/( short1 a, short s )
{
	return make_short1( a.x / s );
}

static __inline__ __host__ __device__ short1 operator/( short s, short1 a )
{
	return make_short1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( short1& a, short s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ short1 clamp( short1 v, short a, short b )
{
	short1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ short1 clamp( short1 v, short1 a, short1 b )
{
	short1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ short1 abs( short1 v )
{
	return make_short1( abs( v.x ) );
}

#endif
