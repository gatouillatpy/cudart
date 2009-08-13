
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACamera.h"

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "CUDADebug.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/***********************************************************************************/

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDACamera::CUDACamera()
	{
		fov = PI * 0.5f;
		ratio = 4.0f / 3.0f;

		zNearPlane = 1.000f;
		zFarPlane = 1999.999f;

		center.x = 0.0f;
		center.y = 0.0f;
		center.z = 0.0f;

		angle.x = PI * 0.5f;
		angle.y = 0.0f;
		angle.z = 0.0f;

		dir.x = 0.0f;
		dir.y = 0.0f;
		dir.z = 1.0f;

		dirty = true;
	}

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	inline void CUDACamera::dirFromAngle()
	{
		if ( angle.x < 0.0f ) angle.x += PI * 2.0f;
		if ( angle.x > PI * 2.0f ) angle.x -= PI * 2.0f;
		if ( angle.y < -PI * 0.5f + 0.001f ) angle.y = -PI * 0.5f + 0.001f;
		if ( angle.y > PI * 0.5f - 0.001f ) angle.y = PI * 0.5f - 0.001f;

		dir.x = cosf( angle.x ) * cosf( angle.y );
		dir.y = sinf( angle.y );
		dir.z = sinf( angle.x ) * cosf( angle.y );

		dir = normalize( dir );
	}

	inline void CUDACamera::angleFromDir()
	{
		dir = normalize( dir );

		angle.y = asinf( dir.y );

		if ( angle.y < -PI * 0.5f + 0.001f ) angle.y = -PI * 0.5f + 0.001f;
		if ( angle.y > PI * 0.5f - 0.001f ) angle.y = PI * 0.5f - 0.001f;

		if ( dir.x > 0 )
			angle.x = asinf( dir.z / cosf( angle.y ) );
		else if ( dir.z > 0 )
			angle.x = acosf( dir.x / cosf( angle.y ) );
		else
			angle.x = acosf( -dir.x / cosf( -angle.y ) ) + PI;

		if ( angle.x < 0.0f ) angle.x += PI * 2.0f;
		if ( angle.x > PI * 2.0f ) angle.x -= PI * 2.0f;
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	void CUDACamera::setCenter( float _x, float _y, float _z )
	{
		center = make_float3( _x, _y, _z );

		dirty = true;
	}

	void CUDACamera::setCenter( float3 _center )
	{
		center = _center;

		dirty = true;
	}

	void CUDACamera::setAngle( float _x, float _y, float _z )
	{
		angle = make_float3( _x, _y, _z );

		dirFromAngle();

		dirty = true;
	}

	void CUDACamera::setAngle( float3 _angle )
	{
		angle = _angle;

		dirFromAngle();

		dirty = true;
	}

	void CUDACamera::lookAt( float _x, float _y, float _z )
	{
		dir = make_float3( _x, _y, _z ) - center;

		angleFromDir();

		dirty = true;
	}

	void CUDACamera::lookAt( float3 _point )
	{
		dir = _point - center;

		angleFromDir();

		dirty = true;
	}

	void CUDACamera::translate( float _x, float _y, float _z )
	{
		center.x += _x;
		center.y += _y;
		center.z += _z;

		dirty = true;
	}

	void CUDACamera::rotate( float _x, float _y, float _z )
	{
		angle.x -= _x;
		angle.y -= _y;
		angle.z -= _z;

		dirFromAngle();

		dirty = true;
	}

	void CUDACamera::moveForward( float _k )
	{
		center.x += _k * dir.x;
		center.y += _k * dir.y;
		center.z += _k * dir.z;

		dirty = true;
	}

	void CUDACamera::moveBackward( float _k )
	{
		center.x -= _k * dir.x;
		center.y -= _k * dir.y;
		center.z -= _k * dir.z;

		dirty = true;
	}

	void CUDACamera::moveLeft( float _k )
	{
		center.x -= _k * dir.z;
		center.z += _k * dir.x;

		dirty = true;
	}

	void CUDACamera::moveRight( float _k )
	{
		center.x += _k * dir.z;
		center.z -= _k * dir.x;

		dirty = true;
	}

	void CUDACamera::setFOV( float _fov )
	{
		fov = _fov;
	}

	void CUDACamera::setRatio( float _ratio )
	{
		ratio = _ratio;
	}

	void CUDACamera::setRatio( int _width, int _height )
	{
		ratio =(float)_width /(float)_height;
	}

	float4x4 CUDACamera::getViewMatrix()
	{
		if ( dirty ) update();

		return matView;
	}

	float4x4 CUDACamera::getProjMatrix()
	{
		if ( dirty ) update();

		return matProj;
	}

	float4x4 CUDACamera::getInvViewMatrix()
	{
		if ( dirty ) update();

		return matInvView;
	}

	float4x4 CUDACamera::getInvProjMatrix()
	{
		if ( dirty ) update();

		return matInvProj;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDACamera::update()
	{
		float3 target = center + 4999.999f * dir;
		float3 normal = make_float3( dir.x, 0.0f, dir.y );

		float4x4 matUp;
		matrixRotationAxis( matUp, normal, angle.z );
		
		float4 eye = make_float4( center, 0.0f );
		float4 at = make_float4( target, 0.0f );
		float4 up = matrixTransform( matUp, make_float4( 0.0f, 1.0f, 0.0f, 1.0f ) );

		matrixLookAt( matView, eye, at, up );

		matrixInverse( matInvView, matView );

		matrixPerspectiveFov( matProj, fov, ratio, zNearPlane, zFarPlane );

		matrixInverse( matInvProj, matProj );

		dirty = false;
	}

}