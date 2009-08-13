
#ifndef _CUDA_SHORT2
#define _CUDA_SHORT2

#include "_cuda_short.h"

static __inline__ __host__ __device__ short2 make_short2( short s )
{
	return make_short2( s, s );
}

static __inline__ __host__ __device__ short2 make_short2( float2 a )
{
	return make_short2( short(a.x), short(a.y) );
}

static __inline__ __host__ __device__ short2 min( short2 a, short2 b )
{
	short2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ short2 max( short2 a, short2 b )
{
	short2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ short2 operator-( short2 a )
{
	return make_short2( -a.x, -a.y );
}

static __inline__ __host__ __device__ short2 operator+( short2 a, short2 b )
{
	return make_short2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( short2& a, short2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ short2 operator-( short2 a, short2 b )
{
	return make_short2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( short2& a, short2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ short2 operator*( short2 a, short2 b )
{
	return make_short2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ short2 operator*( short2 a, short s )
{
	return make_short2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ short2 operator*( short s, short2 a )
{
	return make_short2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( short2& a, short s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ short2 operator/( short2 a, short2 b )
{
	return make_short2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ short2 operator/( short2 a, short s )
{
	return make_short2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ short2 operator/( short s, short2 a )
{
	return make_short2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( short2& a, short s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ short2 clamp( short2 v, short a, short b )
{
	short2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ short2 clamp( short2 v, short2 a, short2 b )
{
	short2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ short2 abs( short2 v )
{
	return make_short2( abs( v.x ), abs( v.y ) );
}

#endif
