
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDASurface.h"

using namespace renderkit;

#include "../CUDACommon.h"

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

template <typename unit_type>
__global__ void kernel_fillSurface( unit_type* pixels, int width, int height, unit_type value )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if ( x >= width || y >= height ) return;
	
	// calculate the offset
	int n = y * width + x;

    // color the pixel at (x,y)
    pixels[n] = value;
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

template <typename unit_type>
void CUDA_fillSurface( CUDASurface<unit_type>* surface, unit_type value )
{
	int width = surface->getWidth();
	int height = surface->getHeight();

	dim3 db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
    dim3 dg = dim3( (width + db.x - 1) / db.x, (height + db.y - 1) / db.y );

	unit_type* data = surface->getPointer();

    kernel_fillSurface<unit_type><<<dg,db>>>
	(
		data,
		width,
		height,
		value
	);

	cudaThreadSynchronize();

    _assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_fillSurface() : Unable to execute CUDA kernel." );
}

// force nvcc à générer le code pour chacun des noyaux
// ajouter les lignes correspondant aux types nécessaires

void dummy_fillSurface()
{
 	CUDA_fillSurface<uint>( NULL, 0 );
 	CUDA_fillSurface<float>( NULL, 0.0f );
 	CUDA_fillSurface<float2>( NULL, make_float2( 0.0f ) );
 	CUDA_fillSurface<float4>( NULL, make_float4( 0.0f ) );
}
