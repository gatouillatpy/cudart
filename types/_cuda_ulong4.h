
#ifndef _CUDA_ULONG4
#define _CUDA_ULONG4

#include "_cuda_ulong.h"

static __inline__ __host__ __device__ ulong4 make_ulong4( ulong s )
{
	return make_ulong4( s, s, s, s );
}

static __inline__ __host__ __device__ ulong4 make_ulong4( ulong3 s )
{
	return make_ulong4( s.x, s.y, s.z, 0 );
}

static __inline__ __host__ __device__ ulong4 make_ulong4( ulong3 s, ulong w )
{
	return make_ulong4( s.x, s.y, s.z, w );
}

static __inline__ __host__ __device__ ulong4 make_ulong4( float4 a )
{
	return make_ulong4( ulong(a.x), ulong(a.y), ulong(a.z), ulong(a.w) );
}

static __inline__ __host__ __device__ ulong4 min( ulong4 a, ulong4 b )
{
	ulong4 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );
	r.w = min( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ ulong4 max( ulong4 a, ulong4 b )
{
	ulong4 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );
	r.w = max( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ ulong4 operator+( ulong4 a, ulong4 b )
{
	return make_ulong4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __inline__ __host__ __device__ void operator+=( ulong4& a, ulong4 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__ ulong4 operator-( ulong4 a, ulong4 b )
{
	return make_ulong4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

static __inline__ __host__ __device__ void operator-=( ulong4& a, ulong4 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static __inline__ __host__ __device__ ulong4 operator*( ulong4 a, ulong4 b )
{
	return make_ulong4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

static __inline__ __host__ __device__ ulong4 operator*( ulong4 a, ulong s )
{
	return make_ulong4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ ulong4 operator*( ulong s, ulong4 a )
{
	return make_ulong4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ void operator*=( ulong4& a, ulong s )
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

static __inline__ __host__ __device__ ulong4 operator/( ulong4 a, ulong4 b )
{
	return make_ulong4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

static __inline__ __host__ __device__ ulong4 operator/( ulong4 a, ulong s )
{
	return make_ulong4( a.x / s, a.y / s, a.z / s, a.w / s );
}

static __inline__ __host__ __device__ ulong4 operator/( ulong s, ulong4 a )
{
	return make_ulong4( s / a.x, s / a.y, s / a.z, s / a.w );
}

static __inline__ __host__ __device__ void operator/=( ulong4& a, ulong s )
{
	a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}

static __inline__ __device__ __host__ ulong4 clamp( ulong4 v, ulong a, ulong b )
{
	ulong4 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );
	r.w = clamp( v.w, a, b );

	return r;
}

static __inline__ __device__ __host__ ulong4 clamp( ulong4 v, ulong4 a, ulong4 b )
{
	ulong4 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );
	r.w = clamp( v.w, a.w, b.w );

	return r;
}

#endif
