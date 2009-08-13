
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDARenderSurface.h"
#include "../CUDABuffer.h"

using namespace renderkit;

#include "../CUDACommon.h"

#include <math_constants.h>

/***********************************************************************************/
/** MACROS GPU                                                                    **/
/***********************************************************************************/

inline __device__ uint make_pixel( float4 v )
{
	uint pixel = 0;

	v *= 255.0f;

	v = fminf( v, make_float4( 255.0f ) );
	v = fmaxf( v, make_float4(   0.0f ) );

	pixel |= ((uint)(v.x) & 0xFF) << 16;
	pixel |= ((uint)(v.y) & 0xFF) <<  8;
	pixel |= ((uint)(v.z) & 0xFF) <<  0;
	pixel |= ((uint)(v.w) & 0xFF) << 24;

	return pixel;
}

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <float4, 1, cudaReadModeElementType> tex_origins;
texture <float4, 1, cudaReadModeElementType> tex_directions;

texture <uint, 1, cudaReadModeElementType> tex_faces_id;
texture <float2, 1, cudaReadModeElementType> tex_coords;
texture <float, 1, cudaReadModeElementType> tex_depths;

texture <float4, 1, cudaReadModeElementType> tex_points;
texture <float4, 1, cudaReadModeElementType> tex_normals;
texture <float4, 1, cudaReadModeElementType> tex_colors;
texture <float2, 1, cudaReadModeElementType> tex_texcoords;
texture <uint, 1, cudaReadModeElementType> tex_materials;

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static void beginCUDAShader( CUDARenderSurface<uint>* output, CUDABuffer* input, dim3& db, dim3& dg )
{
	{
		int width = output->getWidth();
		int height = output->getHeight();

		db = dim3( BLOCK_WIDTH, BLOCK_HEIGHT );
		dg = dim3( (width + db.x - 1) / db.x, (height + db.y - 1) / db.y );
	}

	if ( input->getInputOriginSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_origins, input->getInputOriginSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getInputDirectionSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_directions, input->getInputDirectionSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputFaceIdSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces_id, input->getOutputFaceIdSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputCoordSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 0, 0, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_coords, input->getOutputCoordSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputDepthSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_depths, input->getOutputDepthSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputPointSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_points, input->getOutputPointSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputNormalSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_normals, input->getOutputNormalSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputColorSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_colors, input->getOutputColorSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputTexcoordSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 0, 0, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_texcoords, input->getOutputTexcoordSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}

	if ( input->getOutputMaterialSurface() )
	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_materials, input->getOutputMaterialSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "runCUDAShader() : Unable to bind CUDA texture." );
	}
}

static void endCUDAShader()
{
	cudaThreadSynchronize();

    _assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"runCUDAShader() : Unable to execute CUDA kernel." );

	cudaUnbindTexture( tex_materials );

	cudaUnbindTexture( tex_texcoords );
	
	cudaUnbindTexture( tex_colors );

	cudaUnbindTexture( tex_normals );

	cudaUnbindTexture( tex_points );

	cudaUnbindTexture( tex_faces_id );

	cudaUnbindTexture( tex_coords );

	cudaUnbindTexture( tex_depths );

	cudaUnbindTexture( tex_directions );

	cudaUnbindTexture( tex_origins );
}
