
#ifndef _CUDA_CHAR2
#define _CUDA_CHAR2

#include "_cuda_char.h"

static __inline__ __host__ __device__ char2 make_char2( char s )
{
	return make_char2( s, s );
}

static __inline__ __host__ __device__ char2 make_char2( float2 a )
{
	return make_char2( char(a.x), char(a.y) );
}

static __inline__ __host__ __device__ char2 min( char2 a, char2 b )
{
	char2 r;

	r.x = min( a.x, b.x );
	r.y = min( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ char2 max( char2 a, char2 b )
{
	char2 r;

	r.x = max( a.x, b.x );
	r.y = max( a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ char2 operator-( char2 a )
{
	return make_char2( -a.x, -a.y );
}

static __inline__ __host__ __device__ char2 operator+( char2 a, char2 b )
{
	return make_char2( a.x + b.x, a.y + b.y );
}

static __inline__ __host__ __device__ void operator+=( char2& a, char2 b )
{
	a.x += b.x; a.y += b.y;
}

static __inline__ __host__ __device__ char2 operator-( char2 a, char2 b )
{
	return make_char2( a.x - b.x, a.y - b.y );
}

static __inline__ __host__ __device__ void operator-=( char2& a, char2 b )
{
	a.x -= b.x; a.y -= b.y;
}

static __inline__ __host__ __device__ char2 operator*( char2 a, char2 b )
{
	return make_char2( a.x * b.x, a.y * b.y );
}

static __inline__ __host__ __device__ char2 operator*( char2 a, char s )
{
	return make_char2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ char2 operator*( char s, char2 a )
{
	return make_char2( a.x * s, a.y * s );
}

static __inline__ __host__ __device__ void operator*=( char2& a, char s )
{
	a.x *= s; a.y *= s;
}

static __inline__ __host__ __device__ char2 operator/( char2 a, char2 b )
{
	return make_char2( a.x / b.x, a.y / b.y );
}

static __inline__ __host__ __device__ char2 operator/( char2 a, char s )
{
	return make_char2( a.x / s, a.y / s );
}

static __inline__ __host__ __device__ char2 operator/( char s, char2 a )
{
	return make_char2( s / a.x, s / a.y );
}

static __inline__ __host__ __device__ void operator/=( char2& a, char s )
{
	a.x /= s; a.y /= s;
}

static __inline__ __device__ __host__ char2 clamp( char2 v, char a, char b )
{
	char2 r;

	r.x = clamp( v.x, a, b );
	r.y = clamp( v.y, a, b );

	return r;
}

static __inline__ __device__ __host__ char2 clamp( char2 v, char2 a, char2 b )
{
	char2 r;

	r.x = clamp( v.x, a.x, b.x );
	r.y = clamp( v.y, a.y, b.y );

	return r;
}

static __inline__ __host__ __device__ char2 abs( char2 v )
{
	return make_char2( abs( v.x ), abs( v.y ) );
}

#endif
