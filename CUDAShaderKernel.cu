
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "kernels/CUDAShaderCommon.cuh"

#include "CUDAShader.h"

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ static void kernel_shade( uint* pixels, int pitch, int width, int height, float4 back_color, float4 cam_dir )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= width || y >= height ) return;

	int k = y * width + x;
	int n = y * pitch + x;

	float depth = tex1Dfetch( tex_depths, k );

	if ( depth < +CUDART_NORM_HUGE_F )
	{
		float4 point = tex1Dfetch( tex_points, k );
		float4 normal = tex1Dfetch( tex_normals, k );

		cam_dir.w = 0.0f;
		normal.w = 0.0f;

		//float4 point_color = fabs( reflect( dir, normal ) );
		float4 point_color = make_float4( -dot( cam_dir, normal ) );

		pixels[n] = make_pixel( point_color );
	}
	else
	{
		pixels[n] = make_pixel( back_color );
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void runCUDAShader( CUDARenderSurface<uint>* output, CUDABuffer* input, CUDACamera* camera, CUDAShader* shader )
{
	dim3 db, dg;

	beginCUDAShader( output, input, db, dg );

	int pitch = output->getPitch() / sizeof(uint);
	int width = output->getWidth();
	int height = output->getHeight();

	uint* pixels = output->getPointer();

	float4 back_color = shader->getBackColor();

	float4 cam_dir = make_float4( camera->getDir() );

    kernel_shade<<<dg,db>>>
	(
		pixels,
		pitch,
		width,
		height,
		back_color,
		cam_dir
	);
	
	endCUDAShader();
}
