
#ifndef _CUDA_INT4
#define _CUDA_INT4

#include "_cuda_int.h"

static __inline__ __host__ __device__ int4 make_int4( int s )
{
	return make_int4( s, s, s, s );
}

static __inline__ __host__ __device__ int4 make_int4( int3 s )
{
	return make_int4( s.x, s.y, s.z, 0 );
}

static __inline__ __host__ __device__ int4 make_int4( int3 s, int w )
{
	return make_int4( s.x, s.y, s.z, w );
}

static __inline__ __host__ __device__ int4 make_int4( float4 a )
{
	return make_int4( int(a.x), int(a.y), int(a.z), int(a.w) );
}

static __inline__ __host__ __device__ int4 min( int4 a, int4 b )
{
	int4 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );
	r.w = min( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ int4 max( int4 a, int4 b )
{
	int4 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );
	r.w = max( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ int4 operator-( int4 a )
{
	return make_int4( -a.x, -a.y, -a.z, -a.w );
}

static __inline__ __host__ __device__ int4 operator+( int4 a, int4 b )
{
	return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __inline__ __host__ __device__ void operator+=( int4& a, int4 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__ int4 operator-( int4 a, int4 b )
{
	return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

static __inline__ __host__ __device__ void operator-=( int4& a, int4 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static __inline__ __host__ __device__ int4 operator*( int4 a, int4 b )
{
	return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

static __inline__ __host__ __device__ int4 operator*( int4 a, int s )
{
	return make_int4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ int4 operator*( int s, int4 a )
{
	return make_int4( a.x * s, a.y * s, a.z * s, a.w * s );
}

static __inline__ __host__ __device__ void operator*=( int4& a, int s )
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

static __inline__ __host__ __device__ int4 operator/( int4 a, int4 b )
{
	return make_int4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

static __inline__ __host__ __device__ int4 operator/( int4 a, int s )
{
	return make_int4( a.x / s, a.y / s, a.z / s, a.w / s );
}

static __inline__ __host__ __device__ int4 operator/( int s, int4 a )
{
	return make_int4( s / a.x, s / a.y, s / a.z, s / a.w );
}

static __inline__ __host__ __device__ void operator/=( int4& a, int s )
{
	a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}

static __inline__ __device__ __host__ int4 clamp( int4 v, int a, int b )
{
	int4 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );
	r.w = clamp( v.w, a, b );

	return r;
}

static __inline__ __device__ __host__ int4 clamp( int4 v, int4 a, int4 b )
{
	int4 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );
	r.w = clamp( v.w, a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ int4 abs( int4 v )
{
	return make_int4( abs( v.x ), abs( v.y ), abs( v.z ), abs( v.w ) );
}

#endif
