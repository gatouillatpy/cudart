
#ifndef _CUDA_UINT3
#define _CUDA_UINT3

#include "_cuda_uint.h"

static __inline__ __host__ __device__ uint3 make_uint3( uint s )
{
	return make_uint3( s, s, s );
}

static __inline__ __host__ __device__ uint3 make_uint3( uint2 s )
{
	return make_uint3( s.x, s.y, 0 );
}

static __inline__ __host__ __device__ uint3 make_uint3( uint2 s, uint w )
{
	return make_uint3( s.x, s.y, w );
}

static __inline__ __host__ __device__ uint3 make_uint3( float3 a )
{
	return make_uint3( uint(a.x), uint(a.y), uint(a.z) );
}

static __inline__ __host__ __device__ uint3 min( uint3 a, uint3 b )
{
	uint3 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ uint3 max( uint3 a, uint3 b )
{
	uint3 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ uint3 operator+( uint3 a, uint3 b )
{
	return make_uint3( a.x + b.x, a.y + b.y, a.z + b.z );
}

static __inline__ __host__ __device__ void operator+=( uint3& a, uint3 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

static __inline__ __host__ __device__ uint3 operator-( uint3 a, uint3 b )
{
	return make_uint3( a.x - b.x, a.y - b.y, a.z - b.z );
}

static __inline__ __host__ __device__ void operator-=( uint3& a, uint3 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

static __inline__ __host__ __device__ uint3 operator*( uint3 a, uint3 b )
{
	return make_uint3( a.x * b.x, a.y * b.y, a.z * b.z );
}

static __inline__ __host__ __device__ uint3 operator*( uint3 a, uint s )
{
	return make_uint3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ uint3 operator*( uint s, uint3 a )
{
	return make_uint3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ void operator*=( uint3& a, uint s )
{
	a.x *= s; a.y *= s; a.z *= s;
}

static __inline__ __host__ __device__ uint3 operator/( uint3 a, uint3 b )
{
	return make_uint3( a.x / b.x, a.y / b.y, a.z / b.z );
}

static __inline__ __host__ __device__ uint3 operator/( uint3 a, uint s )
{
	return make_uint3( a.x / s, a.y / s, a.z / s );
}

static __inline__ __host__ __device__ uint3 operator/( uint s, uint3 a )
{
	return make_uint3( s / a.x, s / a.y, s / a.z );
}

static __inline__ __host__ __device__ void operator/=( uint3& a, uint s )
{
	a.x /= s; a.y /= s; a.z /= s;
}

static __inline__ __device__ __host__ uint3 clamp( uint3 v, uint a, uint b )
{
	uint3 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );

	return r;
}

static __inline__ __device__ __host__ uint3 clamp( uint3 v, uint3 a, uint3 b )
{
	uint3 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );

	return r;
}

#endif
