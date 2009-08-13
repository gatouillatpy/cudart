
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

texture <float4, 1, cudaReadModeElementType> tex_vertices;

texture <uint4, 1, cudaReadModeElementType> tex_faces;

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define EPSILON 0.000001f

#define CACHE_SIZE 192

#define BLOCK_COUNT_X 2
#define BLOCK_COUNT_Y 2

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 4

#define BLOCK_SIZE 32

/***********************************************************************************/
/** MACROS GPU                                                                    **/
/***********************************************************************************/

static __inline__ __device__ __host__ bool intersectRectRect( const float4& min0, const float4& max0, const float2& min1, const float2& max1 )
{
	return ( min0.x < max1.x && max0.x > min1.x && min0.y < max1.y && max0.y > min1.y );
}

static __inline__ __device__ __host__ bool intersectTriPoint( const float2& point, const float4& vert0, const float4& vert1, const float4& vert2, float& t, float& u, float& v )
{
	float det = (vert1.x - vert0.x) * (vert2.y - vert0.y) - (vert2.x - vert0.x) * (vert1.y - vert0.y);

	if ( det > -EPSILON && det < EPSILON ) return false;

	float inv_det = 1.0f / det;

	u = ((vert2.x - point.x) * (vert0.y - point.y) - (vert0.x - point.x) * (vert2.y - point.y)) * inv_det;

	if ( u < 0.0f || u > 1.0f ) return false;

	v = ((vert0.x - point.x) * (vert1.y - point.y) - (vert1.x - point.x) * (vert0.y - point.y)) * inv_det;

	if ( v < 0.0f || u + v > 1.0f ) return false;

	t = (1.0f - u - v) * vert0.z + u * vert1.z + v * vert2.z;

	return true;
}

/***********************************************************************************/
/** NOYAUX                                                                        **/
/***********************************************************************************/

__global__ void static kernel_rasterizeMeshTree( uint* faces_id, float2* coords, float* depths,
														float4x4 matrix, int face_count, int width, int height )
{
	int n = threadIdx.y * blockDim.x + threadIdx.x;

	float2 zone_min_id = make_float2( threadIdx.x + 0, threadIdx.y + 0 );
	float2 zone_max_id = make_float2( threadIdx.x + 1, threadIdx.y + 1 );

	__shared__ float4 s_vert0[CACHE_SIZE];
	__shared__ float4 s_vert1[CACHE_SIZE];
	__shared__ float4 s_vert2[CACHE_SIZE];

	__shared__ int s_bit[CACHE_SIZE];

	__shared__ int2 s_box_min[CACHE_SIZE];
	__shared__ int2 s_box_max[CACHE_SIZE];

	int2 zone_count = make_int2( 1 + (width - 1) / BLOCK_WIDTH, 1 + (height - 1) / BLOCK_HEIGHT );
	float2 zone_scale = 0.5f * make_float2( zone_count );

	for ( int k = 0 ; k < face_count ; k += CACHE_SIZE )
	{
		// chaque thread s'occupe de mettre en cache son lot de triangles
		for ( int i = n ; i < CACHE_SIZE ; i += BLOCK_SIZE )
		{
			float4 vert_in, vert_out;

			if ( k + i < face_count )
			{
				uint4 face = tex1Dfetch( tex_faces, k + i );

				float4 tri_min = make_float4( +CUDART_NORM_HUGE_F );
				float4 tri_max = make_float4( -CUDART_NORM_HUGE_F );

				{
					vert_in = tex1Dfetch( tex_vertices, face.x );
					vert_in.w = 1.0f;

					matrixTransform( vert_out, matrix, vert_in );
					vert_out /= fabs( vert_out.w );
					vert_out.w = 0.0f;
					s_vert0[i] = vert_out;

					tri_min = fminf( tri_min, vert_out );
					tri_max = fmaxf( tri_max, vert_out );
				}

				{
					vert_in = tex1Dfetch( tex_vertices, face.y );
					vert_in.w = 1.0f;

					matrixTransform( vert_out, matrix, vert_in );
					vert_out /= fabs( vert_out.w );
					vert_out.w = 0.0f;
					s_vert1[i] = vert_out;

					tri_min = fminf( tri_min, vert_out );
					tri_max = fmaxf( tri_max, vert_out );
				}

				{
					vert_in = tex1Dfetch( tex_vertices, face.z );
					vert_in.w = 1.0f;

					matrixTransform( vert_out, matrix, vert_in );
					vert_out /= fabs( vert_out.w );
					vert_out.w = 0.0f;
					s_vert2[i] = vert_out;

					tri_min = fminf( tri_min, vert_out );
					tri_max = fmaxf( tri_max, vert_out );
				}

				if ( intersectRectRect( tri_min, tri_max, make_float2( -1.0f, -1.0f ), make_float2( +1.0f, +1.0f ) ) )
				{
					s_vert0[i].w = 0.0f;
					s_vert1[i].w = 0.0f;
					s_vert2[i].w = 0.0f;

					s_bit[i] = 1;

					s_box_min[i].x = max( (int)( zone_scale.x * ( tri_min.x + 1.0f ) ), 0 ) / BLOCK_COUNT_X * BLOCK_COUNT_X;
					s_box_min[i].y = max( (int)( zone_scale.y * ( 1.0f - tri_max.y ) ), 0 ) / BLOCK_COUNT_Y * BLOCK_COUNT_Y;
					s_box_max[i].x = min( (int)( zone_scale.x * ( tri_max.x + 1.0f ) ), zone_count.x - 1 );
					s_box_max[i].y = min( (int)( zone_scale.y * ( 1.0f - tri_min.y ) ), zone_count.y - 1 );
				}
				else
				{
					s_bit[i] = 0;
				}
			}
			else
			{
				s_bit[i] = 0;
			}
		}

		__syncthreads();

		for ( int i = 0 ; i < CACHE_SIZE ; i++ )
		{
			if ( s_bit[i] )
			{
				int x0 = s_box_min[i].x;
				int y0 = s_box_min[i].y;
				int x1 = s_box_max[i].x;
				int y1 = s_box_max[i].y;

				float4 pta = s_vert0[i];
				float4 ptb = s_vert1[i];
				float4 ptc = s_vert2[i];

				for ( int j = blockIdx.y + y0 ; j >= y0 && j <= y1 ; j += BLOCK_COUNT_Y )
				{
					for ( int i = blockIdx.x + x0 ; i >= x0 && i <= x1 ; i += BLOCK_COUNT_X )
					{
						int x = i * BLOCK_WIDTH + threadIdx.x;
						int y = j * BLOCK_HEIGHT + threadIdx.y;

						float2 ptw = make_float2
						(
							(float)x * 2.0f / (float)width - 1.0f,
							1.0f - (float)y * 2.0f / (float)height
						);

						float t, u, v;

						if ( intersectTriPoint( ptw, pta, ptb, ptc, t, u, v ) )
						{
							if ( x >= 0 && x < width && y >= 0 && y < height )
							{
								int l = y * width + x;

								if ( t > 0 && t < depths[l] )
								{
									faces_id[l] = k;
									coords[l] = make_float2( u, v );
									depths[l] = t;
								}
							}
						}
					}
				}
			}
		}

		__syncthreads();
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static void bindTextures( CUDAMesh* mesh, CUDAMeshTree* tree )
{
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_vertices, mesh->getVerticesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_rasterizeMeshTree() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces, mesh->getFacesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_rasterizeMeshTree() : Unable to bind CUDA texture." );
	}
}

static void unbindTextures()
{
	cudaUnbindTexture( tex_faces );

	cudaUnbindTexture( tex_vertices );
}

void CUDA_rasterizeMeshTree( CUDABuffer* buffer, CUDACamera* camera,
									CUDAMesh* mesh, CUDAMeshTree* tree )
{
	int width = buffer->getWidth();
	int height = buffer->getHeight();

	float4x4 matrix;
	matrixMultiply( matrix, camera->getViewMatrix(), camera->getProjMatrix() );
	matrixTranspose( matrix );

	int face_count = mesh->getFaceCount();
	int vertex_count = mesh->getVertexCount();

	// enfin on remplit les triangles
	{
		bindTextures( mesh, tree );

		float* depths = (float*)buffer->getOutputDepthSurface()->getPointer();
		float2* coords = (float2*)buffer->getOutputCoordSurface()->getPointer();
		uint* faces_id = (uint*)buffer->getOutputFaceIdSurface()->getPointer();

		dim3 db = dim3( BLOCK_WIDTH, BLOCK_HEIGHT );
		dim3 dg = dim3( BLOCK_COUNT_X, BLOCK_COUNT_Y );

		kernel_rasterizeMeshTree<<<dg,db>>>
		(
			faces_id,
			coords,
			depths,
			matrix,
			face_count,
			width,
			height
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_rasterizeMeshTree() : Unable to execute CUDA kernel." );

		unbindTextures();
	}
}
