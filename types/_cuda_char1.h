
#ifndef _CUDA_CHAR1
#define _CUDA_CHAR1

#include "_cuda_char.h"

static __inline__ __host__ __device__ char1 make_char1( float1 a )
{
	return make_char1( char(a.x) );
}

static __inline__ __host__ __device__ char1 min( char1 a, char1 b )
{
	char1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ char1 max( char1 a, char1 b )
{
	char1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ char1 operator-( char1 a )
{
	return make_char1( -a.x );
}

static __inline__ __host__ __device__ char1 operator+( char1 a, char1 b )
{
	return make_char1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( char1& a, char1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ char1 operator-( char1 a, char1 b )
{
	return make_char1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( char1& a, char1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ char1 operator*( char1 a, char1 b )
{
	return make_char1( a.x * b.x );
}

static __inline__ __host__ __device__ char1 operator*( char1 a, char s )
{
	return make_char1( a.x * s );
}

static __inline__ __host__ __device__ char1 operator*( char s, char1 a )
{
	return make_char1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( char1& a, char s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ char1 operator/( char1 a, char1 b )
{
	return make_char1( a.x / b.x );
}

static __inline__ __host__ __device__ char1 operator/( char1 a, char s )
{
	return make_char1( a.x / s );
}

static __inline__ __host__ __device__ char1 operator/( char s, char1 a )
{
	return make_char1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( char1& a, char s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ char1 clamp( char1 v, char a, char b )
{
	char1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ char1 clamp( char1 v, char1 a, char1 b )
{
	char1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ char1 abs( char1 v )
{
	return make_char1( abs( v.x ) );
}

#endif
