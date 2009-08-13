
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

texture <uint4, 1, cudaReadModeElementType> tex_nodes;

texture <float4, 1, cudaReadModeElementType> tex_rays_orig;
texture <float4, 1, cudaReadModeElementType> tex_rays_dir;

texture <float4, 1, cudaReadModeElementType> tex_vertices;

texture <uint4, 1, cudaReadModeElementType> tex_faces;

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define EPSILON 0.000001f

#define TRIANGLES_MIN 16
#define TRIANGLES_MAX 224

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 4
#define BLOCK_SIZE 32 // BLOCK_WIDTH * BLOCK_HEIGHT

#define STACK_SIZE 96

#define PACKET_BLOCK_WIDTH 16
#define PACKET_BLOCK_HEIGHT 12
#define PACKET_BLOCK_SIZE 192 // PACKET_BLOCK_WIDTH * PACKET_BLOCK_HEIGHT

#define PACKET_STACK_SIZE 128

/***********************************************************************************/
/** MACROS GPU                                                                    **/
/***********************************************************************************/

static __inline__ __device__ bool intersectTriRay( const float4& orig, const float4& dir, const float4& vert0, const float4& edge1, const float4& edge2, float& t, float& u, float& v )
{
	/* begin calculating determinant - also used to calculate U parameter */
	float4 pvec = cross( dir, edge2 );

	/* if determinant is near zero, ray lies in plane of triangle */
	float det = dot( edge1, pvec );

	if ( det > -EPSILON && det < EPSILON ) return 0;

	float inv_det = 1.0f / det;

	/* calculate distance from vert0 to ray origin */
	float4 tvec = orig - vert0;

	/* calculate U parameter and test bounds */
	u = dot( tvec, pvec ) * inv_det;

	if ( u < 0.0f || u > 1.0f ) return false;

	/* prepare to test V parameter */
	float4 qvec = cross( tvec, edge1 );

	/* calculate V parameter and test bounds */
	v = dot( dir, qvec ) * inv_det;

	if ( v < 0.0 || u + v > 1.0f ) return false;

	/* calculate t, ray intersects triangle */
	t = dot( edge2, qvec ) * inv_det;

	return ( t >= 0.0f );
}

static __inline__ __device__ bool intersectBoxRay( const float4& bmin, const float4& bmax, const float4& orig, const float4& invdir, float& t )
{
	float4 v1 = ( bmin - orig ) * invdir;
	float4 v2 = ( bmax - orig ) * invdir;

	float4 vmin = fminf( v1, v2 );
	float4 vmax = fmaxf( v1, v2 );

	float tmin = fmaxf( vmin.z, fmaxf( vmin.x, vmin.y ) );
	float tmax = fminf( vmax.z, fminf( vmax.x, vmax.y ) );

	t = tmin;

	return ( (tmax >= tmin) && (tmax >= 0.0f) );
}

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ static void kernel_raytraceMeshTree( uint* faces_id, float2* coords, float* depths, int width, int height )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float depth;
	float2 coord;
	uint face_id;

	int l = y * width + x;

	float4 orig, dir, invdir;

	if ( x < width && y < height )
	{
		orig = tex1Dfetch( tex_rays_orig, l );
		dir = tex1Dfetch( tex_rays_dir, l );
		invdir = normalize( make_float4( 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z, 0.0f ) );

		depth = depths[l];
	}

	int n = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int s_stack[STACK_SIZE*BLOCK_SIZE];

	int sp, so;

	sp = so = n * STACK_SIZE;

	s_stack[sp++] = 0;

	while ( sp > so )
	{
		__shared__ bvnode node[BLOCK_SIZE];

		{
			int node_id = s_stack[--sp];

			uint4* p = (uint4*)&node[n];

			*p = tex1Dfetch( tex_nodes, 2 * node_id + 0 ); p++;
			*p = tex1Dfetch( tex_nodes, 2 * node_id + 1 );
		}

		if ( isLeaf( node[n] ) )
		{
			int tri_id;
			int tri_count;

			getObjects( node[n], tri_id, tri_count );

			// chaque thread raytrace son rayon
			for ( int k = 0 ; k < tri_count ; k++ )
			{
				float u, v, t;

				uint4 face = tex1Dfetch( tex_faces, tri_id + k );

				float4 vert0 = tex1Dfetch( tex_vertices, face.x );
				float4 edge1 = tex1Dfetch( tex_vertices, face.y ) - vert0;
				float4 edge2 = tex1Dfetch( tex_vertices, face.z ) - vert0;

				vert0.w = 0.0f;
				edge1.w = 0.0f;
				edge2.w = 0.0f;

				if ( intersectTriRay( orig, dir, vert0, edge1, edge2, t, u, v ) )
				{
					if ( t < depth )
					{
						face_id = tri_id + k;
						coord = make_float2( u, v );
						depth = t;
					}
				}
			}
		}
		else
		{
			float t;

			float4 bmin = make_float4( node[n].box_min );
			float4 bmax = make_float4( node[n].box_max );

			if ( intersectBoxRay( bmin, bmax, orig, invdir, t ) )
			{
				if ( t < depth )
				{
					// on evalue l'ordre de parcours des fils
					int near_child_id, far_child_id;
					getChildren( node[n], dir, near_child_id, far_child_id );
			    
					// on push d'abord la boite la plus éloignée, comme on utilise
					// une pile ce sera celle-ci qui sera parcourue le plus tard
					s_stack[sp++] = far_child_id;
					s_stack[sp++] = near_child_id;
				}
			}
		}
	}

	if ( x < width && y < height )
	{
		faces_id[l] = face_id;
		coords[l] = coord;
		depths[l] = depth;
	}
}


__global__ static void kernel_raytraceMeshTreePacket( uint* faces_id, float2* coords, float* depths, int width, int height )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float depth;
	float2 coord;
	uint face_id;

	int l = y * width + x;

	float4 orig, dir, invdir;

	if ( x < width && y < height )
	{
		orig = tex1Dfetch( tex_rays_orig, l );
		dir = tex1Dfetch( tex_rays_dir, l );
		invdir = normalize( make_float4( 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z, 0.0f ) );

		depth = depths[l];
	}

	int n = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int s_stack[PACKET_STACK_SIZE];
	__shared__ int sp, so;

	// le premier thread empile le premier noeud à parcourir
	if ( n == 0 )
	{
		sp = so = 0;

		s_stack[sp++] = 0;
	}

	__syncthreads();

	while ( sp > so )
	{
		__shared__ bvnode node;

		__shared__ bool hit;

		__syncthreads();

		// le premier thread s'occupe seul de copier depuis la mémoire
		// globale vers la mémoire partagée les infos sur le prochain
		// noeud à parcourir
		if ( n == 0 )
		{
			int node_id = s_stack[--sp];

			uint4* p = (uint4*)&node;

			*p = tex1Dfetch( tex_nodes, 2 * node_id + 0 ); p++;
			*p = tex1Dfetch( tex_nodes, 2 * node_id + 1 );

			hit = false;
		}

		__syncthreads();

		if ( isLeaf( node ) )
		{
			int tri_id;
			int tri_count;

			getObjects( node, tri_id, tri_count );

			__shared__ float4 s_vert0[TRIANGLES_MAX];
			__shared__ float4 s_edge1[TRIANGLES_MAX];
			__shared__ float4 s_edge2[TRIANGLES_MAX];

			// chaque thread s'occupe de mettre en cache son lot de triangles
			for ( int k = n ; k < tri_count ; k += PACKET_BLOCK_SIZE )
			{
				uint4 face = tex1Dfetch( tex_faces, tri_id + k );

				s_vert0[k] = tex1Dfetch( tex_vertices, face.x );
				s_edge1[k] = tex1Dfetch( tex_vertices, face.y ) - s_vert0[k];
				s_edge2[k] = tex1Dfetch( tex_vertices, face.z ) - s_vert0[k];

				s_vert0[k].w = 0.0f;
				s_edge1[k].w = 0.0f;
				s_edge2[k].w = 0.0f;
			}

			__syncthreads();

			// puis chaque thread raytrace son rayon
			for ( int k = 0 ; k < tri_count ; k++ )
			{
				float u, v, t;

				float4 vert0 = s_vert0[k];
				float4 edge1 = s_edge1[k];
				float4 edge2 = s_edge2[k];

				if ( intersectTriRay( orig, dir, vert0, edge1, edge2, t, u, v ) )
				{
					if ( t < depth )
					{
						face_id = tri_id + k;
						coord = make_float2( u, v );
						depth = t;
					}
				}
			}

			__syncthreads();
		}
		else
		{
			float t;

			float4 bmin = make_float4( node.box_min );
			float4 bmax = make_float4( node.box_max );

			if ( intersectBoxRay( bmin, bmax, orig, invdir, t ) )
			{
				if ( t < depth )
				{
					hit = true;
				}
			}

			__syncthreads();

			// le premier thread empile les fils si l'un des threads du bloc
			// a vu son rayon intersecter la boite englobante du père
			if ( n == 0 )
			{
				if ( hit )
				{
					// on evalue l'ordre de parcours des fils
					int near_child_id, far_child_id;
					getChildren( node, dir, near_child_id, far_child_id );
			    
					// on push d'abord la boite la plus éloignée, comme on utilise
					// une pile ce sera celle-ci qui sera parcourue le plus tard
					s_stack[sp++] = far_child_id;
					s_stack[sp++] = near_child_id;
				}
			}

			__syncthreads();
		}
	}

	if ( x < width && y < height )
	{
		faces_id[l] = face_id;
		coords[l] = coord;
		depths[l] = depth;
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static void bindTextures( CUDABuffer* buffer, CUDAMesh* mesh, CUDAMeshTree* tree )
{
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_vertices, mesh->getVerticesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_raytraceMeshTree() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_faces, mesh->getFacesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_raytraceMeshTree() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_nodes, tree->getNodesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_raytraceMeshTree() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_rays_orig, buffer->getInputOriginSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_raytraceMeshTree() : Unable to bind CUDA texture." );
	}

	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_rays_dir, buffer->getInputDirectionSurface()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_raytraceMeshTree() : Unable to bind CUDA texture." );
	}
}

static void unbindTextures()
{
	cudaUnbindTexture( tex_rays_orig );

	cudaUnbindTexture( tex_rays_dir );

	cudaUnbindTexture( tex_nodes );

	cudaUnbindTexture( tex_faces );

	cudaUnbindTexture( tex_vertices );
}

void CUDA_raytraceMeshTree( CUDABuffer* buffer, CUDAMesh* mesh, CUDAMeshTree* tree, bool coherency = true )
{
	int width = buffer->getWidth();
	int height = buffer->getHeight();

	bindTextures( buffer, mesh, tree );

	float* depths = (float*)buffer->getOutputDepthSurface()->getPointer();
	float2* coords = (float2*)buffer->getOutputCoordSurface()->getPointer();
	uint* faces_id = (uint*)buffer->getOutputFaceIdSurface()->getPointer();

	if ( coherency )
	{
		dim3 db = dim3( PACKET_BLOCK_WIDTH, PACKET_BLOCK_HEIGHT );
		dim3 dg = dim3( (width + db.x - 1) / db.x, (height + db.y - 1) / db.y );

		kernel_raytraceMeshTreePacket<<<dg,db>>>
		(
			faces_id,
			coords,
			depths,
			width,
			height
		);
	}
	else
	{
		dim3 db = dim3( BLOCK_WIDTH, BLOCK_HEIGHT );
		dim3 dg = dim3( (width + db.x - 1) / db.x, (height + db.y - 1) / db.y );

		kernel_raytraceMeshTree<<<dg,db>>>
		(
			faces_id,
			coords,
			depths,
			width,
			height
		);
	}

	cudaThreadSynchronize();

    _assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
				"CUDA_raytraceMeshTree() : Unable to execute CUDA kernel." );

	unbindTextures();
}
