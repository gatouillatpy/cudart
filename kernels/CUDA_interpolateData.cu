
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDANode.h"
#include "../CUDABox.h"
#include "../CUDAMesh.h"
#include "../CUDAMeshTree.h"
#include "../CUDACamera.h"
#include "../CUDAArray.h"
#include "../CUDASurface.h"
#include "../CUDABuffer.h"

using namespace renderkit;

#include "../CUDACommon.h"

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <uint, 1, cudaReadModeElementType> tex_faces_id;
texture <float2, 1, cudaReadModeElementType> tex_coords;

texture <float4, 1, cudaReadModeElementType> tex_vertices;
texture <float4, 1, cudaReadModeElementType> tex_normals;
texture <float4, 1, cudaReadModeElementType> tex_colors;
texture <float2, 1, cudaReadModeElementType> tex_texcoords;

texture <uint4, 1, cudaReadModeElementType> tex_faces;

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define BLOCK_SIZE 256 // BLOCK_WIDTH * BLOCK_HEIGHT

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ void static kernel_interpolateData( float4* points, float4* normals, float4* colors, float2* texcoords, uint* materials,
																int width, int height, bool hasNormals, bool hasColors, bool hasTexcoords )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x < width && y < height )
	{
		int l = y * width + x;

		uint face_id = tex1Dfetch( tex_faces_id, l );
		float2 coord = tex1Dfetch( tex_coords, l );

		uint4 face = tex1Dfetch( tex_faces, face_id );

		float4 vert0 = tex1Dfetch( tex_vertices, face.x );
		float4 vert1 = tex1Dfetch( tex_vertices, face.y );
		float4 vert2 = tex1Dfetch( tex_vertices, face.z );

		if ( points )
		{
			float4 point = (1 - coord.x - coord.y) * vert0 + coord.x * vert1 + coord.y * vert2;

			points[l] = point;
		}

		if ( normals )
		{
			if ( hasNormals )
			{
				float4 norm0 = tex1Dfetch( tex_normals, face.x );
				float4 norm1 = tex1Dfetch( tex_normals, face.y );
				float4 norm2 = tex1Dfetch( tex_normals, face.z );

				float4 normal = normalize( (1 - coord.x - coord.y) * norm0 + coord.x * norm1 + coord.y * norm2 );

				normals[l] = normal;
			}
			else
			{
				float4 normal = normalize( cross( vert1 - vert0, vert2 - vert0 ) );

				normals[l] = normal;
			}
		}

		if ( colors )
		{
			if ( hasColors )
			{
				float4 color0 = tex1Dfetch( tex_colors, face.x );
				float4 color1 = tex1Dfetch( tex_colors, face.y );
				float4 color2 = tex1Dfetch( tex_colors, face.z );

				float4 color = (1 - coord.x - coord.y) * color0 + coord.x * color1 + coord.y * color2;

				colors[l] = color;
			}
			else
			{
				float4 color = make_float4( 1.0f );

				colors[l] = color;
			}
		}

		if ( texcoords )
		{
			if ( hasTexcoords )
			{
				float2 texc0 = tex1Dfetch( tex_texcoords, face.x );
				float2 texc1 = tex1Dfetch( tex_texcoords, face.y );
				float2 texc2 = tex1Dfetch( tex_texcoords, face.z );

				float2 texcoord = (1 - coord.x - coord.y) * texc0 + coord.x * texc1 + coord.y * texc2;

				texcoords[l] = texcoord;
			}
			else
			{
				float2 texcoord = coord;

				texcoords[l] = texcoord;
			}
		}

		if ( materials )
		{
			uint material = face.w;

			materials[l] = material;
		}
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static void bindTextures( CUDABuffer* buffer, CUDAMesh* mesh )
{
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_vertices, mesh->getVerticesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	if ( mesh->getNormalsArray() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_normals, mesh->getNormalsArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	if ( mesh->getColorsArray() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_colors, mesh->getColorsArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	if ( mesh->getTexcoordsArray() )
	{
		cudaChannelFormatDesc desc = { 32, 32, 0, 0, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_texcoords, mesh->getTexcoordsArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces, mesh->getFacesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces_id, buffer->getOutputFaceIdSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 0, 0, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_coords, buffer->getOutputCoordSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_interpolateData() : Unable to bind CUDA texture." );
	}
}

static void unbindTextures()
{
	cudaUnbindTexture( tex_faces_id );

	cudaUnbindTexture( tex_coords );

	cudaUnbindTexture( tex_faces );

	cudaUnbindTexture( tex_texcoords );

	cudaUnbindTexture( tex_colors );

	cudaUnbindTexture( tex_normals );

	cudaUnbindTexture( tex_vertices );
}

void CUDA_interpolateData( CUDABuffer* buffer, CUDAMesh* mesh )
{
	int width = buffer->getWidth();
	int height = buffer->getHeight();

	bindTextures( buffer, mesh );

	float4* points = buffer->getOutputPointSurface() ? buffer->getOutputPointSurface()->getPointer() : NULL;
	float4* normals = buffer->getOutputNormalSurface() ? buffer->getOutputNormalSurface()->getPointer() : NULL;
	float4* colors = buffer->getOutputColorSurface() ? buffer->getOutputColorSurface()->getPointer() : NULL;
	float2* texcoords = buffer->getOutputTexcoordSurface() ? buffer->getOutputTexcoordSurface()->getPointer() : NULL;
	uint* materials = buffer->getOutputMaterialSurface() ? buffer->getOutputMaterialSurface()->getPointer() : NULL;

	bool hasNormals = (mesh->getNormalsArray() != NULL);
	bool hasColors = (mesh->getColorsArray() != NULL);
	bool hasTexcoords = (mesh->getTexcoordsArray() != NULL);

	dim3 db = dim3( BLOCK_WIDTH, BLOCK_HEIGHT );
	dim3 dg = dim3( (width + db.x - 1) / db.x, (height + db.y - 1) / db.y );

	kernel_interpolateData<<<dg,db>>>
	(
		points,
		normals,
		colors,
		texcoords,
		materials,
		width,
		height,
		hasNormals,
		hasColors,
		hasTexcoords
	);

	cudaThreadSynchronize();

    _assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_interpolateData() : Unable to execute CUDA kernel." );

	unbindTextures();
}
