
#ifndef _CUDA_ULONG2
#define _CUDA_ULONG2

#include "_cuda_ulong.h"

static __inline__ __host__ __device__ ulong2 make_ulong2( ulong s )
{
	return make_ulong2( s, s );
}

static __inline__ __host__ __device__ ulong2 make_ulong2( float2 a )
{
	return make_ulong2( ulong(a.x), ulong(a.y) );
}

static __inline__ __host__ __device__ ulong2 min( ulong2 a, ulong2 b )
{
	ulong2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ ulong2 max( ulong2 a, ulong2 b )
{
	ulong2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ ulong2 operator+( ulong2 a, ulong2 b )
{
	return make_ulong2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( ulong2& a, ulong2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ ulong2 operator-( ulong2 a, ulong2 b )
{
	return make_ulong2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( ulong2& a, ulong2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ ulong2 operator*( ulong2 a, ulong2 b )
{
	return make_ulong2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ ulong2 operator*( ulong2 a, ulong s )
{
	return make_ulong2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ ulong2 operator*( ulong s, ulong2 a )
{
	return make_ulong2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( ulong2& a, ulong s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ ulong2 operator/( ulong2 a, ulong2 b )
{
	return make_ulong2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ ulong2 operator/( ulong2 a, ulong s )
{
	return make_ulong2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ ulong2 operator/( ulong s, ulong2 a )
{
	return make_ulong2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( ulong2& a, ulong s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ ulong2 clamp( ulong2 v, ulong a, ulong b )
{
	ulong2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ ulong2 clamp( ulong2 v, ulong2 a, ulong2 b )
{
	ulong2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

#endif
