
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_SIZE 256

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <uint, 1, cudaReadModeElementType> tex_faces_id;

texture <uint4, 1, cudaReadModeElementType> tex_faces;

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ static void kernel_reorderMesh( uint4* faces_buffer, int face_count )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < face_count )
	{
		uint face_id = tex1Dfetch( tex_faces_id, n );

		uint4 face = tex1Dfetch( tex_faces, face_id );

		faces_buffer[n] = face;
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void CUDA_reorderMeshFaces( uint4* faces, uint* indices, uint face_count )
{
	uint4* faces_buffer; cudaMalloc( (void**)&faces_buffer, face_count * sizeof(uint4) );

	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces_id, indices, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_reorderMeshFaces() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces, faces, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_reorderMeshFaces() : Unable to bind CUDA texture." );
	}

	dim3 db = dim3( BLOCK_SIZE );
	dim3 dg = dim3( ( face_count + db.x - 1 ) / db.x );

	kernel_reorderMesh<<<dg,db>>>
	(
		faces_buffer,
		face_count
	);

	cudaThreadSynchronize();

	_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_reorderMeshFaces() : Unable to execute CUDA kernel." );

	cudaUnbindTexture( tex_faces_id );

	cudaUnbindTexture( tex_faces );

	cudaMemcpy( faces, faces_buffer, face_count * sizeof(uint4), cudaMemcpyDeviceToDevice );

	cudaFree( faces_buffer );
}
