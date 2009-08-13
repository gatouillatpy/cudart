
#ifndef _CUDA_USHORT1
#define _CUDA_USHORT1

#include "_cuda_ushort.h"

static __inline__ __host__ __device__ ushort1 make_ushort1( float1 a )
{
	return make_ushort1( ushort(a.x) );
}

static __inline__ __host__ __device__ ushort1 min( ushort1 a, ushort1 b )
{
	ushort1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ ushort1 max( ushort1 a, ushort1 b )
{
	ushort1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ ushort1 operator-( ushort1 a )
{
	return make_ushort1( -a.x );
}

static __inline__ __host__ __device__ ushort1 operator+( ushort1 a, ushort1 b )
{
	return make_ushort1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( ushort1& a, ushort1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ ushort1 operator-( ushort1 a, ushort1 b )
{
	return make_ushort1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( ushort1& a, ushort1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ ushort1 operator*( ushort1 a, ushort1 b )
{
	return make_ushort1( a.x * b.x );
}

static __inline__ __host__ __device__ ushort1 operator*( ushort1 a, ushort s )
{
	return make_ushort1( a.x * s );
}

static __inline__ __host__ __device__ ushort1 operator*( ushort s, ushort1 a )
{
	return make_ushort1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( ushort1& a, ushort s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ ushort1 operator/( ushort1 a, ushort1 b )
{
	return make_ushort1( a.x / b.x );
}

static __inline__ __host__ __device__ ushort1 operator/( ushort1 a, ushort s )
{
	return make_ushort1( a.x / s );
}

static __inline__ __host__ __device__ ushort1 operator/( ushort s, ushort1 a )
{
	return make_ushort1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( ushort1& a, ushort s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ ushort1 clamp( ushort1 v, ushort a, ushort b )
{
	ushort1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ ushort1 clamp( ushort1 v, ushort1 a, ushort1 b )
{
	ushort1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

#endif
