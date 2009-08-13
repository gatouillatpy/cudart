
#ifndef _CUDA_UCHAR2
#define _CUDA_UCHAR2

#include "_cuda_uchar.h"

static __inline__ __host__ __device__ uchar2 make_uchar2( uchar s )
{
	return make_uchar2( s, s );
}

static __inline__ __host__ __device__ uchar2 make_uchar2( float2 a )
{
	return make_uchar2( uchar(a.x), uchar(a.y) );
}

static __inline__ __host__ __device__ uchar2 min( uchar2 a, uchar2 b )
{
	uchar2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ uchar2 max( uchar2 a, uchar2 b )
{
	uchar2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ uchar2 operator-( uchar2 a )
{
	return make_uchar2( -a.x, -a.y );
}

static __inline__ __host__ __device__ uchar2 operator+( uchar2 a, uchar2 b )
{
	return make_uchar2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( uchar2& a, uchar2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ uchar2 operator-( uchar2 a, uchar2 b )
{
	return make_uchar2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( uchar2& a, uchar2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ uchar2 operator*( uchar2 a, uchar2 b )
{
	return make_uchar2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ uchar2 operator*( uchar2 a, uchar s )
{
	return make_uchar2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ uchar2 operator*( uchar s, uchar2 a )
{
	return make_uchar2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( uchar2& a, uchar s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ uchar2 operator/( uchar2 a, uchar2 b )
{
	return make_uchar2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ uchar2 operator/( uchar2 a, uchar s )
{
	return make_uchar2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ uchar2 operator/( uchar s, uchar2 a )
{
	return make_uchar2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( uchar2& a, uchar s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ uchar2 clamp( uchar2 v, uchar a, uchar b )
{
	uchar2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ uchar2 clamp( uchar2 v, uchar2 a, uchar2 b )
{
	uchar2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

#endif
