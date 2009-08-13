
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#pragma warning (disable : 279)

#include "../CUDACamera.h"
#include "../CUDASurface.h"
#include "../CUDABuffer.h"

using namespace renderkit;

#include "../CUDACommon.h"

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ void kernel_calcPrimaryRays( float4* rays_orig, float4* rays_dir, int width, int height,
											 float4 rayOrig, float4 rayDir, float4 rayX, float4 rayY )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
       
	if ( x >= width || y >= height ) return;

	int n = y * width + x;

	rays_orig[n] = rayOrig;
	rays_dir[n] = rayDir - (float)x * rayX - (float)y * rayY;
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void CUDA_calcPrimaryRays( CUDABuffer* buffer, CUDACamera* camera )
{
	int width = buffer->getWidth();
	int height = buffer->getHeight();

	float y = tanf( camera->getFOV() * 0.5f );
	float x = y * camera->getRatio();

	float4 camDir = make_float4( cos(camera->pitch()) * cos(camera->yaw()), sin(camera->yaw()), sin(camera->pitch()) * cos(camera->yaw()), 0.0f );
	float4 camUp = make_float4( cos(camera->pitch()) * sin(camera->roll()), cos(camera->roll()), 0.0f, 0.0f );

	float4 right = normalize( cross( camDir, camUp ) );
	float4 up = normalize( cross( right, camDir ) );

	float4 rayOrig = make_float4( camera->getCenter() );
	float4 rayDir = y * up + x * right + camDir;
	float4 rayX = 2.0f * x * right / (float)width;
	float4 rayY = 2.0f * y * up / (float)height;

    dim3 Db = dim3( 16, 16 );
    dim3 Dg = dim3( (width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y );

	float4* rays_orig = (float4*)buffer->getInputOriginSurface()->getPointer();
	float4* rays_dir = (float4*)buffer->getInputDirectionSurface()->getPointer();

	kernel_calcPrimaryRays<<<Dg,Db>>>
	(
		rays_orig, rays_dir,
		width, height,
		rayOrig, rayDir, rayX, rayY
	);

	cudaThreadSynchronize();

    _assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_calcPrimaryRays() : Unable to execute CUDA kernel." );
}
