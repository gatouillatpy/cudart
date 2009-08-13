
#ifndef _CUDA_UCHAR1
#define _CUDA_UCHAR1

#include "_cuda_uchar.h"

static __inline__ __host__ __device__ uchar1 make_uchar1( float1 a )
{
	return make_uchar1( uchar(a.x) );
}

static __inline__ __host__ __device__ uchar1 min( uchar1 a, uchar1 b )
{
	uchar1 r;

	r.x = min( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ uchar1 max( uchar1 a, uchar1 b )
{
	uchar1 r;

	r.x = max( a.x, b.x );

	return r;
}

static __inline__ __host__ __device__ uchar1 operator-( uchar1 a )
{
	return make_uchar1( -a.x );
}

static __inline__ __host__ __device__ uchar1 operator+( uchar1 a, uchar1 b )
{
	return make_uchar1( a.x + b.x );
}

static __inline__ __host__ __device__ void operator+=( uchar1& a, uchar1 b )
{
	a.x += b.x;
}

static __inline__ __host__ __device__ uchar1 operator-( uchar1 a, uchar1 b )
{
	return make_uchar1( a.x - b.x );
}

static __inline__ __host__ __device__ void operator-=( uchar1& a, uchar1 b )
{
	a.x -= b.x;
}

static __inline__ __host__ __device__ uchar1 operator*( uchar1 a, uchar1 b )
{
	return make_uchar1( a.x * b.x );
}

static __inline__ __host__ __device__ uchar1 operator*( uchar1 a, uchar s )
{
	return make_uchar1( a.x * s );
}

static __inline__ __host__ __device__ uchar1 operator*( uchar s, uchar1 a )
{
	return make_uchar1( a.x * s );
}

static __inline__ __host__ __device__ void operator*=( uchar1& a, uchar s )
{
	a.x *= s;
}

static __inline__ __host__ __device__ uchar1 operator/( uchar1 a, uchar1 b )
{
	return make_uchar1( a.x / b.x );
}

static __inline__ __host__ __device__ uchar1 operator/( uchar1 a, uchar s )
{
	return make_uchar1( a.x / s );
}

static __inline__ __host__ __device__ uchar1 operator/( uchar s, uchar1 a )
{
	return make_uchar1( s / a.x );
}

static __inline__ __host__ __device__ void operator/=( uchar1& a, uchar s )
{
	a.x /= s;
}

static __inline__ __device__ __host__ uchar1 clamp( uchar1 v, uchar a, uchar b )
{
	uchar1 r;

	r.x = clamp( v.x, a, b );

	return r;
}

static __inline__ __device__ __host__ uchar1 clamp( uchar1 v, uchar1 a, uchar1 b )
{
	uchar1 r;

	r.x = clamp( v.x, a.x, b.x );

	return r;
}

#endif
