
#ifndef _CUDA_INT1
#define _CUDA_INT1

#include "_cuda_int.h"

static __inline__ __host__ __device__ int1 make_int1( float1 a )
{
	return make_int1( int(a.x) );
}

static __inline__ __host__ __device__ int1 min( int1 a, int1 b )
{
	int1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ int1 max( int1 a, int1 b )
{
	int1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ int1 operator-( int1 a )
{
	return make_int1( -a.x );
}

static __inline__ __host__ __device__ int1 operator+( int1 a, int1 b )
{
	return make_int1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( int1& a, int1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ int1 operator-( int1 a, int1 b )
{
	return make_int1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( int1& a, int1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ int1 operator*( int1 a, int1 b )
{
	return make_int1( a.x * b.x );
}

static __inline__ __host__ __device__ int1 operator*( int1 a, int s )
{
	return make_int1( a.x * s );
}

static __inline__ __host__ __device__ int1 operator*( int s, int1 a )
{
	return make_int1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( int1& a, int s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ int1 operator/( int1 a, int1 b )
{
	return make_int1( a.x / b.x );
}

static __inline__ __host__ __device__ int1 operator/( int1 a, int s )
{
	return make_int1( a.x / s );
}

static __inline__ __host__ __device__ int1 operator/( int s, int1 a )
{
	return make_int1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( int1& a, int s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ int1 clamp( int1 v, int a, int b )
{
	int1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ int1 clamp( int1 v, int1 a, int1 b )
{
	int1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ int1 abs( int1 v )
{
	return make_int1( abs( v.x ) );
}

#endif
