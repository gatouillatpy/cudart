
#ifndef _CUDA_FLOAT4
#define _CUDA_FLOAT4

#include "_cuda_float.h"

static __inline__ __host__ __device__ float4 make_float4( float s )
{
	return make_float4( s, s, s, s );
}

static __inline__ __host__ __device__ float4 make_float4( float3 a )
{
	return make_float4( a.x, a.y, a.z, 0.0f );
}

static __inline__ __host__ __device__ float4 make_float4( float3 a, float w )
{
	return make_float4( a.x, a.y, a.z, w );
}

static __inline__ __host__ __device__ float4 make_float4( int4 a )
{
	return make_float4( float(a.x), float(a.y), float(a.z), float(a.w) );
}

static __inline__ __host__ __device__ float4 operator-( float4& a )
{
	return make_float4( -a.x, -a.y, -a.z, -a.w );
}

static __inline__ __host__ __device__ float4 fminf( float4 a, float3 b )
{
	float4 r;

	r.x = fminf( a.x, b.x );
	r.y = fminf( a.y, b.y );
	r.z = fminf( a.z, b.z );
	r.w = a.w;

	return r;
}

static __inline__ __host__ __device__ float4 fmaxf( float4 a, float3 b )
{
	float4 r;

	r.x = fmaxf( a.x, b.x );
	r.y = fmaxf( a.y, b.y );
	r.z = fmaxf( a.z, b.z );
	r.w = a.w;

	return r;
}

static __inline__ __host__ __device__ float4 fminf( float4 a, float4 b )
{
	float4 r;

	r.x = fminf( a.x, b.x );
	r.y = fminf( a.y, b.y );
	r.z = fminf( a.z, b.z );
	r.w = fminf( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ float4 fmaxf( float4 a, float4 b )
{
	float4 r;

	r.x = fmaxf( a.x, b.x );
	r.y = fmaxf( a.y, b.y );
	r.z = fmaxf( a.z, b.z );
	r.w = fmaxf( a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ float4 operator+( float4 a, float4 b )
{
	return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

static __inline__ __host__ __device__ void operator+=( float4& a, float4 b )
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__ float4 operator-( float4 a, float4 b )
{
	return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

static __inline__ __host__ __device__ void operator-=( float4& a, float4 b )
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static __inline__ __host__ __device__ float4 operator*( float4 a, float4 b )
{
	return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

static __inline__ __host__ __device__ float4 operator*( float4 a, float s )
{
	return make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}
static __inline__ __host__ __device__ float4 operator*( float s, float4 a )
{
	return make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}
static __inline__ __host__ __device__ void operator*=( float4& a, float s )
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

static __inline__ __host__ __device__ float4 operator/( float4 a, float4 b )
{
	return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

static __inline__ __host__ __device__ float4 operator/( float4 a, float s )
{
	float inv = 1.0f / s;

	return a * inv;
}

static __inline__ __host__ __device__ float4 operator/( float s, float4 a )
{
	return make_float4( s / a.x, s / a.y, s / a.z, s / a.w );
}

static __inline__ __host__ __device__ void operator/=( float4& a, float s )
{
	float inv = 1.0f / s;

	a *= inv;
}

static __inline__ __device__ __host__ float4 lerp( float4 a, float4 b, float t )
{
	return a + t * ( b - a );
}

static __inline__ __device__ __host__ float4 clamp( float4 v, float a, float b )
{
	float4 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );
	r.z = clamp( v.z, a, b );
	r.w = clamp( v.w, a, b );

	return r;
}

static __inline__ __device__ __host__ float4 clamp( float4 v, float4 a, float4 b )
{
	float4 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );
	r.z = clamp( v.z, a.z, b.z );
	r.w = clamp( v.w, a.w, b.w );

	return r;
}

static __inline__ __host__ __device__ float dot( float4 a, float4 b )
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __inline__ __host__ __device__ float length( float4 r )
{
	return sqrtf( dot( r, r ) );
}

static __inline__ __host__ __device__ float4 normalize( float4 v )
{
	float inv_len = rsqrtf( dot( v, v ) );

	return v * inv_len;
}

static __inline__ __host__ __device__ float4 floor( float4 v )
{
	return make_float4( floor(v.x), floor(v.y), floor(v.z), floor(v.w) );
}

static __inline__ __host__ __device__ float4 fabs( float4 v )
{
	return make_float4( fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w) );
}

static __inline__ __host__ __device__ float4 cross( float4 a, float4 b )
{
	float4 r;

	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;
	r.w = 0.0f;

	return r;
}

static __inline__ __host__ __device__ float4 cross( float4 a, float4 b, float4 c )
{
	float4 r;

	float xy = b.x * c.y - c.x * b.y;
	float xz = b.x * c.z - c.x * b.z;
	float xw = b.x * c.w - c.x * b.w;
	float yz = b.y * c.z - c.y * b.z;
	float yw = b.y * c.w - c.y * b.w;
	float zw = b.z * c.w - c.z * b.w;

	r.x = a.y * zw - a.z * yw + a.w * yz;
	r.y = a.z * xw - a.x * zw - a.w * xz;
	r.z = a.x * yw - a.y * xw + a.w * xy;
	r.w = a.y * xz - a.x * yz - a.z * xy;

	return r;
}

static __inline__ __host__ __device__ float4 reflect( float4 i, float4 n )
{
	return i - 2.0f * n * dot( n, i );
}

#endif
