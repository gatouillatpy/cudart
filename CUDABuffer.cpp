
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDABuffer.h"

#include <vector_types.h>
#include <math_constants.h>

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
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

template <typename unit_type>
void CUDA_fillSurface
(
	renderkit::CUDASurface<unit_type>* surface,
	unit_type value
);

/***********************************************************************************/

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDABuffer::CUDABuffer( int _width, int _height )
	{
		_assert( _width > 0, __FILE__, __LINE__, "CUDABuffer::CUDABuffer() : Invalid parameter '_width', must be positive." );
		_assert( _height > 0, __FILE__, __LINE__, "CUDABuffer::CUDABuffer() : Invalid parameter '_height', must be positive." );

		width = _width;
		height = _height;

		in_origins = new CUDASurface<float4>( width, height );
		in_directions = new CUDASurface<float4>( width, height );

		out_faces_id = new CUDASurface<uint>( width, height );
		out_coords = new CUDASurface<float2>( width, height );
		out_depths = new CUDASurface<float>( width, height );

		out_points = NULL;
		out_normals = NULL;
		out_colors = NULL;
		out_texcoords = NULL;
		out_materials = NULL;
	}

	CUDABuffer::~CUDABuffer()
	{
		if ( out_depths )
			delete out_depths;

		if ( out_coords )
			delete out_coords;

		if ( out_faces_id )
			delete out_faces_id;

		if ( out_materials )
			delete out_materials;

		if ( out_texcoords )
			delete out_texcoords;

		if ( out_colors )
			delete out_colors;

		if ( out_normals )
			delete out_normals;

		if ( out_points )
			delete out_points;

		if ( in_directions )
			delete in_directions;

		if ( in_origins )
			delete in_origins;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDABuffer::clearInputSurfaces()
	{
		CUDA_fillSurface<float4>( in_origins, make_float4( 0.0f ) );

		CUDA_fillSurface<float4>( in_directions, make_float4( 0.0f ) );
	}

	void CUDABuffer::clearOutputSurfaces()
	{
		CUDA_fillSurface<uint>( out_faces_id, 0 );

		CUDA_fillSurface<float2>( out_coords, make_float2( 0.0f ) );

		CUDA_fillSurface<float>( out_depths, +CUDART_NORM_HUGE_F );

		if ( out_materials )
			CUDA_fillSurface<uint>( out_materials, 0 );

		if ( out_texcoords )
			CUDA_fillSurface<float2>( out_texcoords, make_float2( 0.0f ) );

		if ( out_colors )
			CUDA_fillSurface<float4>( out_colors, make_float4( 0.0f ) );

		if ( out_normals )
			CUDA_fillSurface<float4>( out_normals, make_float4( 0.0f ) );

		if ( out_points )
			CUDA_fillSurface<float4>( out_points, make_float4( 0.0f ) );
	}

}