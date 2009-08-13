
// Code original : MX^ADD (mxadd@mxadd.org)

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BUBBLE_THREAD_LIMIT 512

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <uint, 1, cudaReadModeElementType> tex_keys;
texture <uint, 1, cudaReadModeElementType> tex_indices;

/***********************************************************************************/
/** NOYAUX                                                                        **/
/***********************************************************************************/

__global__ static void kernel_bubbleSort( uint* keys, uint* indices, uint length )
{
	__shared__ uint s_keys[BUBBLE_THREAD_LIMIT];
	__shared__ uint s_indices[BUBBLE_THREAD_LIMIT];

	const uint n = threadIdx.x;

	if ( length <= 1 ) return;

	// chaque thread copie un élément 
	s_keys[n] = tex1Dfetch( tex_keys, n );
	s_indices[n] = tex1Dfetch( tex_indices, n );

	__syncthreads();

	{
		const uint k = (n << 1);
		const uint j = (length << 1);

		uint ka, kb;
		uint ia, ib;

		for ( uint i = 0 ; i < j ; i++ )
		{
			if ( ( i & 0x01 ) == 0 )
			{
				// [0|1] [2|3] [4|5] [6|7] ...
				if ( ( k + 1 ) < length )
				{
					ka = s_keys[k+0];
					kb = s_keys[k+1];

					ia = s_indices[k+0];
					ib = s_indices[k+1];

					if ( ka > kb )
					{
						s_keys[k+0] = kb;
						s_keys[k+1] = ka;

						s_indices[k+0] = ib;
						s_indices[k+1] = ia;
					}
				}
			}
			else
			{
				// [1|2] [3|4] [5|6] [7|8] ...
				if ( ( k + 2 ) < length )
				{
					ka = s_keys[k+1];
					kb = s_keys[k+2];

					ia = s_indices[k+1];
					ib = s_indices[k+2];

					if ( ka > kb )
					{
						s_keys[k+1] = kb;
						s_keys[k+2] = ka;

						s_indices[k+1] = ib;
						s_indices[k+2] = ia;
					}
				}
			}

			__syncthreads();
		}
	}

	keys[n] = s_keys[n];
	indices[n] = s_indices[n];
}

__global__ static void kernel_bubbleSortInit( uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
		indices[n] = n;
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void CUDA_bubbleSort512( uint* keys, uint* indices, uint length, bool init_indices = false )
{
	_assert( length <= BUBBLE_THREAD_LIMIT, __FILE__, __LINE__,
				"CUDA_bubbleSort512() : Unable to execute CUDA kernel." );

	// on initialise les indices
	if ( init_indices )
	{
		dim3 db = dim3( min( length, BUBBLE_THREAD_LIMIT ) );
		dim3 dg = dim3( 1 );

		kernel_bubbleSortInit<<<dg,db>>>
		(
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bubbleSort512() : Unable to execute CUDA kernel." );
	}

	// on monte les textures pour les données en entrée
	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_keys, keys, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_bubbleSort512() : Unable to bind CUDA texture." );

		if ( cudaBindTexture( NULL, tex_indices, indices, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_bubbleSort512() : Unable to bind CUDA texture." );
	}

	// on effectue le tri
	{
		dim3 db = dim3( min( length, BUBBLE_THREAD_LIMIT ) );
		dim3 dg = dim3( 1 );

		kernel_bubbleSort<<<dg,db>>>
		(
			keys,
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bubbleSort() : Unable to execute CUDA kernel." );
	}

	// on démonte les textures
	{
		cudaUnbindTexture( tex_indices );
		cudaUnbindTexture( tex_keys );
	}
}
