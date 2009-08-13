
#ifndef _CUDA_USHORT3
#define _CUDA_USHORT3

#include "_cuda_ushort.h"

static __inline__ __host__ __device__ ushort3 make_ushort3( ushort s )
{
	return make_ushort3( s, s, s );
}

static __inline__ __host__ __device__ ushort3 make_ushort3( ushort2 s )
{
	return make_ushort3( s.x, s.y, 0 );
}

static __inline__ __host__ __device__ ushort3 make_ushort3( ushort2 s, ushort w )
{
	return make_ushort3( s.x, s.y, w );
}

static __inline__ __host__ __device__ ushort3 make_ushort3( float3 a )
{
	return make_ushort3( ushort(a.x), ushort(a.y), ushort(a.z) );
}

static __inline__ __host__ __device__ ushort3 min( ushort3 a, ushort3 b )
{
	ushort3 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );
	r.z = min( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ ushort3 max( ushort3 a, ushort3 b )
{
	ushort3 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );
	r.z = max( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ ushort3 operator-( ushort3 a )
{
	return make_ushort3( -a.x, -a.y, -a.z );
}

static __inline__ __host__ __device__ ushort3 operator+( ushort3 a, ushort3 b )
{
	return make_ushort3( a.x + b.x, a.y + b.y, a.z + b.z );
}

static __inline__ __host__ __device__ void operator+=( ushort3& a, ushort3 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

static __inline__ __host__ __device__ ushort3 operator-( ushort3 a, ushort3 b )
{
	return make_ushort3( a.x - b.x, a.y - b.y, a.z - b.z );
}

static __inline__ __host__ __device__ void operator-=( ushort3& a, ushort3 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

static __inline__ __host__ __device__ ushort3 operator*( ushort3 a, ushort3 b )
{
	return make_ushort3( a.x * b.x, a.y * b.y, a.z * b.z );
}

static __inline__ __host__ __device__ ushort3 operator*( ushort3 a, ushort s )
{
	return make_ushort3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ ushort3 operator*( ushort s, ushort3 a )
{
	return make_ushort3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ void operator*=( ushort3& a, ushort s )
{
	a.x *= s; a.y *= s; a.z *= s;
}

static __inline__ __host__ __device__ ushort3 operator/( ushort3 a, ushort3 b )
{
	return make_ushort3( a.x / b.x, a.y / b.y, a.z / b.z );
}

static __inline__ __host__ __device__ ushort3 operator/( ushort3 a, ushort s )
{
	return make_ushort3( a.x / s, a.y / s, a.z / s );
}

static __inline__ __host__ __device__ ushort3 operator/( ushort s, ushort3 a )
{
	return make_ushort3( s / a.x, s / a.y, s / a.z );
}

static __inline__ __host__ __device__ void operator/=( ushort3& a, ushort s )
{
	a.x /= s; a.y /= s; a.z /= s;
}

static __inline__ __device__ __host__ ushort3 clamp( ushort3 v, ushort a, ushort b )
{
	ushort3 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );

	return r;
}

static __inline__ __device__ __host__ ushort3 clamp( ushort3 v, ushort3 a, ushort3 b )
{
	ushort3 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );

	return r;
}

#endif
