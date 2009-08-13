
#ifndef _CUDA_FLOAT4X4
#define _CUDA_FLOAT4X4

#include "_cuda_float4.h"

// ATTENTION : Toutes les fonctions relatives aux matrices de type float4x4,
//             contrairement à celles de type MAT44, renvoient des données
//             stockées selon l'ordre row-major.

struct float4x4
{
	float4 r0;
	float4 r1;
	float4 r2;
	float4 r3;
};

static __inline__ __host__ __device__ void matrixDeterminant( float& r, const float4x4& m )
{
	r  = m.r0.x * m.r1.y * m.r2.z * m.r3.w;
	r += m.r0.x * m.r1.z * m.r2.w * m.r3.y;
	r += m.r0.x * m.r1.w * m.r2.y * m.r3.z;

	r += m.r0.y * m.r1.x * m.r2.w * m.r3.z;
	r += m.r0.y * m.r1.z * m.r2.x * m.r3.w;
	r += m.r0.y * m.r1.w * m.r2.z * m.r3.x;

	r += m.r0.z * m.r1.x * m.r2.y * m.r3.w;
	r += m.r0.z * m.r1.y * m.r2.w * m.r3.x;
	r += m.r0.z * m.r1.w * m.r2.x * m.r3.y;

	r += m.r0.w * m.r1.x * m.r2.z * m.r3.y;
	r += m.r0.w * m.r1.y * m.r2.x * m.r3.z;
	r += m.r0.w * m.r1.z * m.r2.y * m.r3.x;

	r -= m.r0.x * m.r1.y * m.r2.w * m.r3.z;
	r -= m.r0.x * m.r1.z * m.r2.y * m.r3.w;
	r -= m.r0.x * m.r1.w * m.r2.z * m.r3.y;

	r -= m.r0.y * m.r1.x * m.r2.z * m.r3.w;
	r -= m.r0.y * m.r1.z * m.r2.w * m.r3.x;
	r -= m.r0.y * m.r1.w * m.r2.x * m.r3.z;

	r -= m.r0.z * m.r1.x * m.r2.w * m.r3.y;
	r -= m.r0.z * m.r1.y * m.r2.x * m.r3.w;
	r -= m.r0.z * m.r1.w * m.r2.y * m.r3.x;

	r -= m.r0.w * m.r1.x * m.r2.y * m.r3.z;
	r -= m.r0.w * m.r1.y * m.r2.z * m.r3.x;
	r -= m.r0.w * m.r1.z * m.r2.x * m.r3.y;
}

static __inline__ __host__ __device__ float matrixDeterminant( const float4x4& m )
{
	float r;

	matrixDeterminant( r, m );

	return r;
}

static __inline__ __host__ __device__ void matrixTranspose( float4x4& r, const float4x4& m )
{
	r.r0 = make_float4( m.r0.x, m.r1.x, m.r2.x, m.r3.x );
	r.r1 = make_float4( m.r0.y, m.r1.y, m.r2.y, m.r3.y );
	r.r2 = make_float4( m.r0.z, m.r1.z, m.r2.z, m.r3.z );
	r.r3 = make_float4( m.r0.w, m.r1.w, m.r2.w, m.r3.w );
}

static __inline__ __host__ __device__ float4x4 matrixTranspose( const float4x4& m )
{
	float4x4 r;

	matrixTranspose( r, m );

	return r;
}

static __inline__ __host__ __device__ void matrixTranspose( float4x4& r )
{
	float4x4 m = r;

	matrixTranspose( r, m );
}

static __inline__ __host__ __device__ void matrixTransposeInverse( float4x4& r, const float4x4& m )
{
	float det;

	matrixDeterminant( det, m );

	if ( det == 0.0f ) return;

	float inv_det = 1.0f / det;

	r.r0 = +inv_det * cross( m.r1, m.r2, m.r3 );
	r.r1 = -inv_det * cross( m.r0, m.r2, m.r3 );
	r.r2 = +inv_det * cross( m.r0, m.r1, m.r3 );
	r.r3 = -inv_det * cross( m.r0, m.r1, m.r2 );
}

static __inline__ __host__ __device__ float4x4 matrixTransposeInverse( const float4x4& m )
{
	float4x4 r;

	matrixTransposeInverse( r, m );

	return r;
}

static __inline__ __host__ __device__ void matrixTransposeInverse( float4x4& r )
{
	float4x4 m = r;

	matrixTransposeInverse( r, m );
}

static __inline__ __host__ __device__ void matrixInverse( float4x4& r, const float4x4& m )
{
	matrixTransposeInverse( r, m );

	matrixTranspose( r );
}

static __inline__ __host__ __device__ float4x4 matrixInverse( const float4x4& m )
{
	float4x4 r;

	matrixInverse( r, m );

	return r;
}

static __inline__ __host__ __device__ void matrixInverse( float4x4& r )
{
	float4x4 m = r;

	matrixInverse( r, m );
}

static __inline__ __host__ __device__ void matrixMultiply( float4x4& r, const float4x4& a, const float4x4& b )
{
	float4x4 t;
	
	matrixTranspose( t, b );

	r.r0.x = dot( a.r0, t.r0 );
	r.r0.y = dot( a.r0, t.r1 );
	r.r0.z = dot( a.r0, t.r2 );
	r.r0.w = dot( a.r0, t.r3 );

	r.r1.x = dot( a.r1, t.r0 );
	r.r1.y = dot( a.r1, t.r1 );
	r.r1.z = dot( a.r1, t.r2 );
	r.r1.w = dot( a.r1, t.r3 );

	r.r2.x = dot( a.r2, t.r0 );
	r.r2.y = dot( a.r2, t.r1 );
	r.r2.z = dot( a.r2, t.r2 );
	r.r2.w = dot( a.r2, t.r3 );

	r.r3.x = dot( a.r3, t.r0 );
	r.r3.y = dot( a.r3, t.r1 );
	r.r3.z = dot( a.r3, t.r2 );
	r.r3.w = dot( a.r3, t.r3 );
}

static __inline__ __host__ __device__ float4x4 matrixMultiply( const float4x4& a, const float4x4& b )
{
	float4x4 r;

	matrixMultiply( r, a, b );

	return r;
}

static __inline__ __host__ __device__ float4x4 operator*( const float4x4& a, const float4x4& b )
{
	return matrixMultiply( a, b );
}

static __inline__ __host__ __device__ void matrixTransform( float4& r, const float4x4& m, const float4& v )
{
	r.x = dot( m.r0, v );
	r.y = dot( m.r1, v );
	r.z = dot( m.r2, v );
	r.w = dot( m.r3, v );
}

static __inline__ __host__ __device__ float4 matrixTransform( const float4x4& m, const float4& v )
{
	float4 r;

	matrixTransform( r, m, v );

	return r;
}

static __inline__ __host__ __device__ void matrixTransformCoord( float4& r, const float4x4& m, float4 v )
{
	v.w = 1.0f;

	r.x = dot( m.r0, v );
	r.y = dot( m.r1, v );
	r.z = dot( m.r2, v );
	r.w = dot( m.r3, v );
}

static __inline__ __host__ __device__ float4 matrixTransformCoord( const float4x4& m, const float4& v )
{
	float4 r;

	matrixTransformCoord( r, m, v );

	return r;
}

static __inline__ __host__ __device__ float4 operator*( const float4& v, const float4x4& m )
{
	return matrixTransform( m, v );
}

static __inline__ __host__ __device__ float4 operator*( const float4x4& m, const float4& v )
{
	return matrixTransform( m, v );
}

static __inline__ __host__ __device__ void matrixIdentity( float4x4& r )
{
	r.r0 = make_float4( 1.0f, 0.0f, 0.0f, 0.0f );
	r.r1 = make_float4( 0.0f, 1.0f, 0.0f, 0.0f );
	r.r2 = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixIdentity()
{
	float4x4 r;

	matrixIdentity( r );
	
	return r;
}

static __inline__ __host__ __device__ void matrixTranslation( float4x4& r, const float x, const float y, const float z )
{
	r.r0 = make_float4( 1.0f, 0.0f, 0.0f,    x );
	r.r1 = make_float4( 0.0f, 1.0f, 0.0f,    y );
	r.r2 = make_float4( 0.0f, 0.0f, 1.0f,    z );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixTranslation( const float x, const float y, const float z )
{
	float4x4 r;

	matrixTranslation( r, x, y, z );

	return r;
}

static __inline__ __host__ __device__ void matrixRotationX( float4x4& r, const float angle )
{
	float u = cosf( angle );
	float v = sinf( angle );

	r.r0 = make_float4( 1.0f, 0.0f, 0.0f, 0.0f );
	r.r1 = make_float4( 0.0f,    u,   -v, 0.0f );
	r.r2 = make_float4( 0.0f,    v,    u, 0.0f );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixRotationX( const float angle )
{
	float4x4 r;

	matrixRotationX( r, angle );

	return r;
}

static __inline__ __host__ __device__ void matrixRotationY( float4x4& r, const float angle )
{
	float u = cosf( angle );
	float v = sinf( angle );

	r.r0 = make_float4(    u, 0.0f,    v, 0.0f );
	r.r1 = make_float4( 0.0f, 1.0f, 0.0f, 0.0f );
	r.r2 = make_float4(   -v, 0.0f,    u, 0.0f );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixRotationY( const float angle )
{
	float4x4 r;

	matrixRotationY( r, angle );

	return r;
}

static __inline__ __host__ __device__ void matrixRotationZ( float4x4& r, const float angle )
{
	float u = cosf( angle );
	float v = sinf( angle );

	r.r0 = make_float4(    u,   -v, 0.0f, 0.0f );
	r.r1 = make_float4(    v,    u, 0.0f, 0.0f );
	r.r2 = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixRotationZ( const float angle )
{
	float4x4 r;

	matrixRotationZ( r, angle );

	return r;
}

static __inline__ __host__ __device__ void matrixRotationAxis( float4x4& r, const float3& axis, const float angle )
{
	float u = cosf( angle );
	float v = sinf( angle );
	float w = 1.0f - u;

	float3 n = normalize( axis );

	r.r0.x = w * n.x * n.x + u;
	r.r0.y = w * n.x * n.y - v * n.z;
	r.r0.z = w * n.x * n.z + v * n.y;
	r.r0.w = 0.0f;

	r.r1.x = w * n.y * n.x + v * n.z;
	r.r1.y = w * n.y * n.y + u;
	r.r1.z = w * n.y * n.z - v * n.x;
	r.r1.w = 0.0f;

	r.r2.x = w * n.z * n.x - v * n.y;
	r.r2.y = w * n.z * n.y + v * n.x;
	r.r2.z = w * n.z * n.z + u;
	r.r2.w = 0.0f;

	r.r3.x = 0.0f;
	r.r3.y = 0.0f;
	r.r3.z = 0.0f;
	r.r3.w = 1.0f;
}

static __inline__ __host__ __device__ float4x4 matrixRotationAxis( const float3& axis, const float angle )
{
	float4x4 r;

	matrixRotationAxis( r, axis, angle );

	return r;
}

static __inline__ __host__ __device__ void matrixRotationYawPitchRoll( float4x4& r, const float yaw, const float pitch, const float roll )
{
	float4x4 matPitch, matYaw, matRoll;

	matrixRotationX( matPitch, pitch );
	matrixRotationY( matYaw, yaw );
	matrixRotationZ( matRoll, roll );

	r = matPitch * matYaw * matRoll;
}

static __inline__ __host__ __device__ float4x4 matrixRotationYawPitchRoll( const float yaw, const float pitch, const float roll )
{
	float4x4 r;

	matrixRotationYawPitchRoll( r, yaw, pitch, roll );

	return r;
}

static __inline__ __host__ __device__ void matrixScale( float4x4& r, const float x, const float y, const float z )
{
	r.r0 = make_float4(    x, 0.0f, 0.0f, 0.0f );
	r.r1 = make_float4( 0.0f,    y, 0.0f, 0.0f );
	r.r2 = make_float4( 0.0f, 0.0f,    z, 0.0f );
	r.r3 = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
}

static __inline__ __host__ __device__ float4x4 matrixScale( const float x, const float y, const float z )
{
	float4x4 r;

	matrixScale( r, x, y, z );

	return r;
}

static __inline__ __host__ __device__ void matrixPerspectiveFov( float4x4& r, const float fov, const float ratio, const float zNear, const float zFar )
{
	float yScale = 1.0f / tanf( fov * 0.5f );
	float xScale = yScale / ratio;

	float zFactor = zFar / ( zFar - zNear );
	float wFactor = -zNear * zFactor;

	r.r0 = make_float4( xScale,   0.0f,    0.0f, 0.0f );
	r.r1 = make_float4(   0.0f, yScale,    0.0f, 0.0f );
	r.r2 = make_float4(   0.0f,   0.0f, zFactor, 1.0f );
	r.r3 = make_float4(   0.0f,   0.0f, wFactor, 0.0f );
}

static __inline__ __host__ __device__ void matrixLookAt( float4x4& r, const float4& eye, const float4& at, const float4& up )
{
	float4 zAxis = normalize( at - eye );
	float4 xAxis = normalize( cross( up, zAxis ) );
	float4 yAxis = cross( zAxis, xAxis );

	float xFactor = -dot( xAxis, eye );
	float yFactor = -dot( yAxis, eye );
	float zFactor = -dot( zAxis, eye );

	r.r0 = make_float4( xAxis.x, yAxis.x, zAxis.x, 0.0f );
	r.r1 = make_float4( xAxis.y, yAxis.y, zAxis.y, 0.0f );
	r.r2 = make_float4( xAxis.z, yAxis.z, zAxis.z, 0.0f );
	r.r3 = make_float4( xFactor, yFactor, zFactor, 1.0f );
}

#endif
