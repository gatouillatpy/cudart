
#ifndef _CUDA_USHORT4
#define _CUDA_USHORT4

#include "_cuda_ushort.h"

static __inline__ __host__ __device__ ushort4 make_ushort4( ushort s )
{
	return make_ushort4( s, s, s, s );
}

static __inline__ __host__ __device__ ushort4 make_ushort4( ushort3 s )
{
	return make_ushort4( s.x, s.y, s.z, 0 );
}

static __inline__ __host__ __device__ ushort4 make_ushort4( ushort3 s, ushort w )
{
	return make_ushort4( s.x, s.y, s.z, w );
}

static __inline__ __host__ __device__ ushort4 make_ushort4( float4 a )
{
	return make_ushort4( ushort(a.x), ushort(a.y), ushort(a.z), ushort(a.w) );
}

static __inline__ __host__ __device__ ushort4 min( ushort4 a, ushort4 b )
{
	ushort4 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );
	r.w = min( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ ushort4 max( ushort4 a, ushort4 b )
{
	ushort4 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );
	r.w = max( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ ushort4 operator-( ushort4 a )
{
	return make_ushort4( -a.x, -a.y, -a.z, -a.w );
}

static __inline__ __host__ __device__ ushort4 operator+( ushort4 a, ushort4 b )
{
	return make_ushort4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __inline__ __host__ __device__ void operator+=( ushort4& a, ushort4 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__ ushort4 operator-( ushort4 a, ushort4 b )
{
	return make_ushort4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

static __inline__ __host__ __device__ void operator-=( ushort4& a, ushort4 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static __inline__ __host__ __device__ ushort4 operator*( ushort4 a, ushort4 b )
{
	return make_ushort4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

static __inline__ __host__ __device__ ushort4 operator*( ushort4 a, ushort s )
{
	return make_ushort4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ ushort4 operator*( ushort s, ushort4 a )
{
	return make_ushort4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ void operator*=( ushort4& a, ushort s )
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

static __inline__ __host__ __device__ ushort4 operator/( ushort4 a, ushort4 b )
{
	return make_ushort4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

static __inline__ __host__ __device__ ushort4 operator/( ushort4 a, ushort s )
{
	return make_ushort4( a.x / s, a.y / s, a.z / s, a.w / s );
}

static __inline__ __host__ __device__ ushort4 operator/( ushort s, ushort4 a )
{
	return make_ushort4( s / a.x, s / a.y, s / a.z, s / a.w );
}

static __inline__ __host__ __device__ void operator/=( ushort4& a, ushort s )
{
	a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}

static __inline__ __device__ __host__ ushort4 clamp( ushort4 v, ushort a, ushort b )
{
	ushort4 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );
	r.w = clamp( v.w, a, b );

	return r;
}

static __inline__ __device__ __host__ ushort4 clamp( ushort4 v, ushort4 a, ushort4 b )
{
	ushort4 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );
	r.w = clamp( v.w, a.w, b.w );

	return r;
}

#endif
