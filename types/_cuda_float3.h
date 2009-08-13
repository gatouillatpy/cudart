
#ifndef _CUDA_FLOAT3
#define _CUDA_FLOAT3

#include "_cuda_float.h"

static __inline__ __host__ __device__ float3 make_float3( float s )
{
	return make_float3( s, s, s );
}

static __inline__ __host__ __device__ float3 make_float3( float2 a )
{
	return make_float3( a.x, a.y, 0.0f );
}

static __inline__ __host__ __device__ float3 make_float3( float2 a, float s )
{
	return make_float3( a.x, a.y, s );
}

static __inline__ __host__ __device__ float3 make_float3( float4 a )
{
	return make_float3( a.x, a.y, a.z );
}

static __inline__ __host__ __device__ float3 make_float3( int3 a )
{
	return make_float3( float(a.x), float(a.y), float(a.z) );
}

static __inline__ __host__ __device__ float3 operator-( float3& a )
{
	return make_float3( -a.x, -a.y, -a.z );
}

static __inline__ __host__ __device__ float3 fminf( float3 a, float4 b )
{
	float3 r;

	r.x = fminf( a.x, b.x );
	r.y = fminf( a.y, b.y );
	r.z = fminf( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ float3 fmaxf( float3 a, float4 b )
{
	float3 r;

	r.x = fmaxf( a.x, b.x );
	r.y = fmaxf( a.y, b.y );
	r.z = fmaxf( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ float3 fminf( float3 a, float3 b )
{
	float3 r;

	r.x = fminf( a.x, b.x );
	r.y = fminf( a.y, b.y );
	r.z = fminf( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ float3 fmaxf( float3 a, float3 b )
{
	float3 r;

	r.x = fmaxf( a.x, b.x );
	r.y = fmaxf( a.y, b.y );
	r.z = fmaxf( a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ float3 operator+( float3 a, float3 b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

static __inline__ __host__ __device__ float3 operator+( float3 a, float b )
{
	return make_float3( a.x + b, a.y + b, a.z + b );
}

static __inline__ __host__ __device__ void operator+=( float3& a, float3 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

static __inline__ __host__ __device__ float3 operator-( float3 a, float3 b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

static __inline__ __host__ __device__ float3 operator-( float3 a, float b )
{
	return make_float3( a.x - b, a.y - b, a.z - b );
}

static __inline__ __host__ __device__ void operator-=( float3& a, float3 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

static __inline__ __host__ __device__ float3 operator*( float3 a, float3 b )
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

static __inline__ __host__ __device__ float3 operator*( float3 a, float s )
{
	return make_float3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ float3 operator*( float s, float3 a )
{
	return make_float3( a.x * s, a.y * s, a.z * s );
}

static __inline__ __host__ __device__ void operator*=( float3& a, float s )
{
	a.x *= s; a.y *= s; a.z *= s;
}

static __inline__ __host__ __device__ float3 operator/( float3 a, float3 b )
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
}

static __inline__ __host__ __device__ float3 operator/( float3 a, float s )
{
	float inv = 1.0f / s;

	return a * inv;
}

static __inline__ __host__ __device__ float3 operator/( float s, float3 a )
{
	return make_float3( s / a.x, s / a.y, s / a.z );
}

static __inline__ __host__ __device__ void operator/=( float3& a, float s )
{
	float inv = 1.0f / s;

	a *= inv;
}

static __inline__ __device__ __host__ float3 lerp( float3 a, float3 b, float t )
{
	return a + t * ( b - a );
}

static __inline__ __device__ __host__ float3 clamp( float3 v, float a, float b )
{
	float3 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );

	return r;
}

static __inline__ __device__ __host__ float3 clamp( float3 v, float3 a, float3 b )
{
	float3 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );

	return r;
}

static __inline__ __host__ __device__ float dot( float3 a, float3 b )
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __inline__ __host__ __device__ float3 cross( float3 a, float3 b )
{ 
	float3 r;

	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;

	return r;
}

static __inline__ __host__ __device__ float length( float3 v )
{
	return sqrtf( dot( v, v ) );
}

static __inline__ __host__ __device__ float3 normalize( float3 v )
{
	float inv_len = rsqrtf( dot( v, v ) );

	return v * inv_len;
}

static __inline__ __host__ __device__ float3 floor( float3 v )
{
	return make_float3( floor( v.x ), floor( v.y ), floor( v.z ) );
}

static __inline__ __host__ __device__ float3 reflect( float3 i, float3 n )
{
	return i - 2.0f * n * dot( n, i );
}

static __inline__ __host__ __device__ float3 fabs( float3 v )
{
	return make_float3( fabs( v.x ), fabs( v.y ), fabs( v.z ) );
}

#endif
