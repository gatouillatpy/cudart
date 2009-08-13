
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDABox.h"

using namespace renderkit;

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_SIZE 256

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <uint4, 1, cudaReadModeElementType> tex_faces;

texture <float4, 1, cudaReadModeElementType> tex_vertices;

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ static void kernel_initIndices( uint* indices, uint count )
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if ( k < count )
		indices[k] = k;
}

__global__ static void kernel_calcBoxes( aabox* boxes, int count )
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int n = threadIdx.x;

	__shared__ aabox s_boxes[BLOCK_SIZE];

	if ( k < count )
	{
		uint4 face = tex1Dfetch( tex_faces, k );

		float4 vert0 = tex1Dfetch( tex_vertices, face.x );
		float4 vert1 = tex1Dfetch( tex_vertices, face.y );
		float4 vert2 = tex1Dfetch( tex_vertices, face.z );

		s_boxes[n].reset( vert0 );
		s_boxes[n].merge( vert1 );
		s_boxes[n].merge( vert2 );

		boxes[k] = s_boxes[n];
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void CUDA_calcMeshBoxes( aabox* boxes, uint* indices, uint4* faces, float4* vertices, uint count )
{
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces, faces, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_calcMeshBoxes() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_vertices, vertices, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_calcMeshBoxes() : Unable to bind CUDA texture." );
	}

	dim3 db = dim3( BLOCK_SIZE );
	dim3 dg = dim3( ( count + db.x - 1 ) / db.x );

	kernel_initIndices<<<dg,db>>>
	(
		indices,
		count
	);

	kernel_calcBoxes<<<dg,db>>>
	(
		boxes,
		count
	);

	cudaThreadSynchronize();

	_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_calcMeshBoxes() : Unable to execute CUDA kernel." );

	cudaUnbindTexture( tex_vertices );

	cudaUnbindTexture( tex_faces );
}
