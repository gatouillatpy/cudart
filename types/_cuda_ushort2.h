
#ifndef _CUDA_USHORT2
#define _CUDA_USHORT2

#include "_cuda_ushort.h"

static __inline__ __host__ __device__ ushort2 make_ushort2( ushort s )
{
	return make_ushort2( s, s );
}

static __inline__ __host__ __device__ ushort2 make_ushort2( float2 a )
{
	return make_ushort2( ushort(a.x), ushort(a.y) );
}

static __inline__ __host__ __device__ ushort2 min( ushort2 a, ushort2 b )
{
	ushort2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ ushort2 max( ushort2 a, ushort2 b )
{
	ushort2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ ushort2 operator-( ushort2 a )
{
	return make_ushort2( -a.x, -a.y );
}

static __inline__ __host__ __device__ ushort2 operator+( ushort2 a, ushort2 b )
{
	return make_ushort2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( ushort2& a, ushort2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ ushort2 operator-( ushort2 a, ushort2 b )
{
	return make_ushort2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( ushort2& a, ushort2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ ushort2 operator*( ushort2 a, ushort2 b )
{
	return make_ushort2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ ushort2 operator*( ushort2 a, ushort s )
{
	return make_ushort2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ ushort2 operator*( ushort s, ushort2 a )
{
	return make_ushort2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( ushort2& a, ushort s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ ushort2 operator/( ushort2 a, ushort2 b )
{
	return make_ushort2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ ushort2 operator/( ushort2 a, ushort s )
{
	return make_ushort2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ ushort2 operator/( ushort s, ushort2 a )
{
	return make_ushort2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( ushort2& a, ushort s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ ushort2 clamp( ushort2 v, ushort a, ushort b )
{
	ushort2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ ushort2 clamp( ushort2 v, ushort2 a, ushort2 b )
{
	ushort2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

#endif
