
#ifndef _CUDA_SHORT4
#define _CUDA_SHORT4

#include "_cuda_short.h"

static __inline__ __host__ __device__ short4 make_short4( short s )
{
	return make_short4( s, s, s, s );
}

static __inline__ __host__ __device__ short4 make_short4( short3 s )
{
	return make_short4( s.x, s.y, s.z, 0 );
}

static __inline__ __host__ __device__ short4 make_short4( short3 s, short w )
{
	return make_short4( s.x, s.y, s.z, w );
}

static __inline__ __host__ __device__ short4 make_short4( float4 a )
{
	return make_short4( short(a.x), short(a.y), short(a.z), short(a.w) );
}

static __inline__ __host__ __device__ short4 min( short4 a, short4 b )
{
	short4 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );
	r.w = min( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ short4 max( short4 a, short4 b )
{
	short4 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );
	r.w = max( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ short4 operator-( short4 a )
{
	return make_short4( -a.x, -a.y, -a.z, -a.w );
}

static __inline__ __host__ __device__ short4 operator+( short4 a, short4 b )
{
	return make_short4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __inline__ __host__ __device__ void operator+=( short4& a, short4 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__ short4 operator-( short4 a, short4 b )
{
	return make_short4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

static __inline__ __host__ __device__ void operator-=( short4& a, short4 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static __inline__ __host__ __device__ short4 operator*( short4 a, short4 b )
{
	return make_short4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

static __inline__ __host__ __device__ short4 operator*( short4 a, short s )
{
	return make_short4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ short4 operator*( short s, short4 a )
{
	return make_short4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ void operator*=( short4& a, short s )
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

static __inline__ __host__ __device__ short4 operator/( short4 a, short4 b )
{
	return make_short4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

static __inline__ __host__ __device__ short4 operator/( short4 a, short s )
{
	return make_short4( a.x / s, a.y / s, a.z / s, a.w / s );
}

static __inline__ __host__ __device__ short4 operator/( short s, short4 a )
{
	return make_short4( s / a.x, s / a.y, s / a.z, s / a.w );
}

static __inline__ __host__ __device__ void operator/=( short4& a, short s )
{
	a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}

static __inline__ __device__ __host__ short4 clamp( short4 v, short a, short b )
{
	short4 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );
	r.w = clamp( v.w, a, b );

	return r;
}

static __inline__ __device__ __host__ short4 clamp( short4 v, short4 a, short4 b )
{
	short4 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );
	r.w = clamp( v.w, a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ short4 abs( short4 v )
{
	return make_short4( abs( v.x ), abs( v.y ), abs( v.z ), abs( v.w ) );
}

#endif
