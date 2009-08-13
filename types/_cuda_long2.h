
#ifndef _CUDA_LONG2
#define _CUDA_LONG2

#include "_cuda_long.h"

static __inline__ __host__ __device__ long2 make_long2( long s )
{
	return make_long2( s, s );
}

static __inline__ __host__ __device__ long2 make_long2( float2 a )
{
	return make_long2( long(a.x), long(a.y) );
}

static __inline__ __host__ __device__ long2 min( long2 a, long2 b )
{
	long2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ long2 max( long2 a, long2 b )
{
	long2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ long2 operator-( long2 a )
{
	return make_long2( -a.x, -a.y );
}

static __inline__ __host__ __device__ long2 operator+( long2 a, long2 b )
{
	return make_long2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( long2& a, long2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ long2 operator-( long2 a, long2 b )
{
	return make_long2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( long2& a, long2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ long2 operator*( long2 a, long2 b )
{
	return make_long2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ long2 operator*( long2 a, long s )
{
	return make_long2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ long2 operator*( long s, long2 a )
{
	return make_long2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( long2& a, long s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ long2 operator/( long2 a, long2 b )
{
	return make_long2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ long2 operator/( long2 a, long s )
{
	return make_long2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ long2 operator/( long s, long2 a )
{
	return make_long2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( long2& a, long s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ long2 clamp( long2 v, long a, long b )
{
	long2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ long2 clamp( long2 v, long2 a, long2 b )
{
	long2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ long2 abs( long2 v )
{
	return make_long2( abs( v.x ), abs( v.y ) );
}

#endif
