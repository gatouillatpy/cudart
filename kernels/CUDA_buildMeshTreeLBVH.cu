
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDAMesh.h"
#include "../CUDAMeshTree.h"
#include "../CUDABin.h"
#include "../CUDANode.h"
#include "../CUDABox.h"
#include "../CUDAArray.h"

#include "../CUDADebug.h"

using namespace renderkit;

#include "../CUDACommon.h"

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void CUDA_calcMeshBoxes( aabox* boxes, uint* indices, uint4* faces, float4* vertices, uint count );

void CUDA_reorderMeshFaces( uint4* faces, uint* indices, uint count );

void CUDA_radixSort( uint* values, uint length, uint bit_depth = 32 );

void CUDA_radixSortGeneral( uint* keys, uint* indices, uint length, uint bit_depth = 32, bool init_indices = false );

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_SIZE 256

#define STACK_SIZE 1024

#define MIN_LEAF_SIZE 16
#define MAX_LEAF_SIZE 224

/***********************************************************************************/
/** MEMOIRE GLOBALE                                                               **/
/***********************************************************************************/

__device__ uint split_count;

__device__ uint level_split_count[32];

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <float4, 1, cudaReadModeElementType> tex_boxes;
texture <uint, 1, cudaReadModeElementType> tex_indices;

texture <uint, 1, cudaReadModeElementType> tex_codes;

texture <uint, 1, cudaReadModeElementType> tex_splits_idx;

/***********************************************************************************/
/** MACROS CPU                                                                    **/
/***********************************************************************************/

inline __host__ int fast_log2( float val )
{
	return ( ( *((int*)&val) & 0x7F800000 ) >> 23 ) - 0x7F;
}

inline __host__ int fast_log2( uint val )
{
	return fast_log2( (float)val );
}

inline __host__ static void host_set_split_count( uint _split_count )
{
	cudaMemcpyToSymbol( "split_count", &_split_count, sizeof(uint), 0, cudaMemcpyHostToDevice );
}

inline __host__ static uint host_get_split_count()
{
	uint _split_count; cudaMemcpyFromSymbol( &_split_count, "split_count", sizeof(uint), 0, cudaMemcpyDeviceToHost ); return _split_count;
}

inline __host__ static bvnode make_node( const int axis, const int child_id )
{
	bvnode bvn;

	bvn.node.type = 0;
	bvn.node.axis = axis;
	bvn.node.child_id = child_id;
	bvn.box_min = make_float3( +CUDART_NORM_HUGE_F);
	bvn.box_max = make_float3( -CUDART_NORM_HUGE_F);

	return bvn;
}

inline __host__ static bvnode make_leaf( const int obj_id, const int obj_count )
{
	bvnode bvn;

	bvn.leaf.type = 1;
	bvn.leaf.obj_count = obj_count;
	bvn.leaf.obj_id = obj_id;
	bvn.box_min = make_float3( +CUDART_NORM_HUGE_F);
	bvn.box_max = make_float3( -CUDART_NORM_HUGE_F);

	return bvn;
}

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

__global__ static void kernel_calcMortonCodes( uint* codes, float4 root_box_min, float4 root_box_max, uint box_count )
{
	const uint n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < box_count )
	{
		float4 root_length = root_box_max - root_box_min;

		float root_scale_x = 2047.0f / root_length.x;
		float root_scale_y = 2047.0f / root_length.y;
		float root_scale_z = 1023.0f / root_length.z;

		float4 box_min = tex1Dfetch( tex_boxes, 2 * n + 0 );
		float4 box_max = tex1Dfetch( tex_boxes, 2 * n + 1 );

		float4 center = 0.5f * ( box_min + box_max );

		uint code_x = (uint)( ( center.x - root_box_min.x ) * root_scale_x );
		uint code_y = (uint)( ( center.y - root_box_min.y ) * root_scale_y );
		uint code_z = (uint)( ( center.z - root_box_min.z ) * root_scale_z );

		uint code = 0;

		code |= (code_y & 0x0001) <<  0;
		code |= (code_x & 0x0001) <<  1;
		code |= (code_z & 0x0001) <<  2;
		code |= (code_y & 0x0002) <<  2;
		code |= (code_x & 0x0002) <<  3;
		code |= (code_z & 0x0002) <<  4;
		code |= (code_y & 0x0004) <<  4;
		code |= (code_x & 0x0004) <<  5;
		code |= (code_z & 0x0004) <<  6;
		code |= (code_y & 0x0008) <<  6;
		code |= (code_x & 0x0008) <<  7;
		code |= (code_z & 0x0008) <<  8;
		code |= (code_y & 0x0010) <<  8;
		code |= (code_x & 0x0010) <<  9;
		code |= (code_z & 0x0010) << 10;
		code |= (code_y & 0x0020) << 10;
		code |= (code_x & 0x0020) << 11;
		code |= (code_z & 0x0020) << 12;
		code |= (code_y & 0x0040) << 12;
		code |= (code_x & 0x0040) << 13;
		code |= (code_z & 0x0040) << 14;
		code |= (code_y & 0x0080) << 14;
		code |= (code_x & 0x0080) << 15;
		code |= (code_z & 0x0080) << 16;
		code |= (code_y & 0x0100) << 16;
		code |= (code_x & 0x0100) << 17;
		code |= (code_z & 0x0100) << 18;
		code |= (code_y & 0x0200) << 18;
		code |= (code_x & 0x0200) << 19;
		code |= (code_z & 0x0200) << 20;
		code |= (code_y & 0x0400) << 20;
		code |= (code_x & 0x0400) << 21;

		codes[n] = code;
	}
}

__global__ static void kernel_findSplits( uint* splits_idx, uint* splits_lvl, uint box_count )
{
	const uint n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < box_count - 1 )
	{
		uint code0 = tex1Dfetch( tex_codes, n + 0 );
		uint code1 = tex1Dfetch( tex_codes, n + 1 );

		uint diff = code0 ^ code1;

		if ( diff != 0 )
		{
			uint level;

			     if ( diff & 0x80000000 ) level =  0;
			else if ( diff & 0x40000000 ) level =  1;
			else if ( diff & 0x20000000 ) level =  2;
			else if ( diff & 0x10000000 ) level =  3;
			else if ( diff & 0x08000000 ) level =  4;
			else if ( diff & 0x04000000 ) level =  5;
			else if ( diff & 0x02000000 ) level =  6;
			else if ( diff & 0x01000000 ) level =  7;
			else if ( diff & 0x00800000 ) level =  8;
			else if ( diff & 0x00400000 ) level =  9;
			else if ( diff & 0x00200000 ) level = 10;
			else if ( diff & 0x00100000 ) level = 11;
			else if ( diff & 0x00080000 ) level = 12;
			else if ( diff & 0x00040000 ) level = 13;
			else if ( diff & 0x00020000 ) level = 14;
			else if ( diff & 0x00010000 ) level = 15;
			else if ( diff & 0x00008000 ) level = 16;
			else if ( diff & 0x00004000 ) level = 17;
			else if ( diff & 0x00002000 ) level = 18;
			else if ( diff & 0x00001000 ) level = 19;
			else if ( diff & 0x00000800 ) level = 20;
			else if ( diff & 0x00000400 ) level = 21;
			else if ( diff & 0x00000200 ) level = 22;
			else if ( diff & 0x00000100 ) level = 23;
			else if ( diff & 0x00000080 ) level = 24;
			else if ( diff & 0x00000040 ) level = 25;
			else if ( diff & 0x00000020 ) level = 26;
			else if ( diff & 0x00000010 ) level = 27;
			else if ( diff & 0x00000008 ) level = 28;
			else if ( diff & 0x00000004 ) level = 29;
			else if ( diff & 0x00000002 ) level = 30;
			else if ( diff & 0x00000001 ) level = 31;

			// level_split_count[level]++;
			atomicAdd( &level_split_count[level], 1 );

			// k = split_count++;
			uint k = atomicAdd( &split_count, 1 );

			splits_idx[k] = n + 1;
			splits_lvl[k] = level;
		}
	}
}

__global__ static void kernel_calcBoundingBoxes( bvnode* nodes_buffer, uint node_count, uint node_level )
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;

	if ( k < node_count )
	{
		__shared__ bvnode bvn[BLOCK_SIZE];

		bvn[n] = nodes_buffer[k];

		if ( node_level == 0 && isLeaf( bvn[n] ) )
		{
			aabox node_box;
			node_box.reset();

			uint tri_index = bvn[n].leaf.obj_id;
			uint tri_bound = tri_index + bvn[n].leaf.obj_count;

			for ( uint i = tri_index ; i < tri_bound ; i++ )
			{
				uint box_id = tex1Dfetch( tex_indices, i );

				float4 box_min = tex1Dfetch( tex_boxes, 2 * box_id + 0 );
				float4 box_max = tex1Dfetch( tex_boxes, 2 * box_id + 1 );

				node_box.merge( box_min, box_max );
			}

			bvn[n].box_min = make_float3( node_box.min );
			bvn[n].box_max = make_float3( node_box.max );

			nodes_buffer[k] = bvn[n];
		}
		else if ( node_level != 0 && isNode( bvn[n] ) )
		{
			// si ce noeud n'a pas encore été traité alors
			if ( bvn[n].box_max.z - bvn[n].box_min.z < 0.0f )
			{
				int child_id = bvn[n].node.child_id;

				bvn[n] = nodes_buffer[child_id+0];

				float3 left_child_box_min = bvn[n].box_min;
				float3 left_child_box_max = bvn[n].box_max;

				// si son fils gauche l'a déjà été ...
				if ( left_child_box_max.z - left_child_box_min.z > 0.0f )
				{
					bvn[n] = nodes_buffer[child_id+1];

					float3 right_child_box_min = bvn[n].box_min;
					float3 right_child_box_max = bvn[n].box_max;

					// ... et que son fils droit aussi alors
					if ( right_child_box_max.z - right_child_box_min.z > 0.0f )
					{
						bvn[n] = nodes_buffer[k];

						bvn[n].box_min = fminf( left_child_box_min, right_child_box_min );
						bvn[n].box_max = fmaxf( left_child_box_max, right_child_box_max );

						nodes_buffer[k] = bvn[n];
					}
				}
			}
		}
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static uint createLeaf( uint tri_index, uint tri_bound,
							uint node_level, uint& node_count,
								uint node_id, bvnode* nodes_buffer,
									bool tree_complete, uint& real_depth );

static uint createNode( uint tri_index, uint tri_bound,
							uint node_level, uint& node_count,
								uint node_id, bvnode* nodes_buffer,
									bool tree_complete, uint& real_depth )
{
	if ( tri_bound - tri_index < MAX_LEAF_SIZE )
		return createLeaf( tri_index, tri_bound, node_level,
							node_count, node_id, nodes_buffer, tree_complete, real_depth );

	uint tri_split = (tri_index + tri_bound) >> 1;

	uint left_child_id = ++node_count;
	uint right_child_id = ++node_count;

	nodes_buffer[node_id] = make_node( node_level % 3, left_child_id );

// 	char spaces[120] = {0}; memset( spaces, ' ', node_level * 2 );
// 	printf( "%s [NODE] ID=%d ; TRI_ID=%d ; TRI_BD=%d\n", spaces, node_id, tri_index, tri_bound );

	createNode( tri_index, tri_split, node_level + 1,
					node_count, left_child_id, nodes_buffer, tree_complete, real_depth );
	createNode( tri_split, tri_bound, node_level + 1,
					node_count, right_child_id, nodes_buffer, tree_complete, real_depth );

	return node_id;
}

static uint createLeaf( uint tri_index, uint tri_bound,
							uint node_level, uint& node_count,
								uint node_id, bvnode* nodes_buffer,
									bool tree_complete, uint& real_depth )
{
	if ( tree_complete && (tri_bound - tri_index > MAX_LEAF_SIZE) )
		return createNode( tri_index, tri_bound, node_level, node_count,
										node_id, nodes_buffer, tree_complete, real_depth );													

	if ( node_level > real_depth ) real_depth = node_level;

// 	char spaces[120] = {0}; memset( spaces, ' ', node_level * 2 );
// 	printf( "%s [LEAF] ID=%d ; TRI_ID=%d ; TRI_BD=%d\n", spaces, node_id, tri_index, tri_bound );

	nodes_buffer[node_id] = make_leaf( tri_index, tri_bound - tri_index );

	return node_id;
}

static uint createNode( uint** host_level_splits_idx,
							uint** host_level_splits_ptr,
								uint tri_index, uint tri_bound,
									uint node_level, uint& node_count,
										uint node_id, bvnode* nodes_buffer,
											bool tree_complete, uint tree_depth, uint& real_depth )
{
	if ( tri_bound - tri_index < MIN_LEAF_SIZE )
		return createLeaf( tri_index, tri_bound, node_level,
							node_count, node_id, nodes_buffer, tree_complete, real_depth );

	if ( node_level == tree_depth )
		return createLeaf( tri_index, tri_bound, node_level,
							node_count, node_id, nodes_buffer, tree_complete, real_depth );

	if ( host_level_splits_ptr[node_level] == host_level_splits_idx[node_level+1] )
		return createLeaf( tri_index, tri_bound, node_level,
							node_count, node_id, nodes_buffer, tree_complete, real_depth );

	uint tri_split;
	
	while ( true )
	{
		tri_split = *host_level_splits_ptr[node_level];

		if ( tri_split >= tri_bound )
			return createLeaf( tri_index, tri_bound, node_level,
								node_count, node_id, nodes_buffer, tree_complete, real_depth );

		if ( tri_split > tri_index ) break;

		host_level_splits_ptr[node_level]++;
	}

	uint left_child_id = ++node_count;
	uint right_child_id = ++node_count;

// 	char spaces[120] = {0}; memset( spaces, ' ', node_level * 2 );
// 	printf( "%s [NODE] ID=%d ; TRI_ID=%d ; TRI_BD=%d\n", spaces, node_id, tri_index, tri_bound );

	nodes_buffer[node_id] = make_node( node_level % 3, left_child_id );

	createNode( host_level_splits_idx, host_level_splits_ptr,
					tri_index, tri_split, node_level + 1, node_count,
							left_child_id, nodes_buffer, tree_complete, tree_depth, real_depth );

	createNode( host_level_splits_idx, host_level_splits_ptr,
					tri_split, tri_bound, node_level + 1, node_count,
							right_child_id, nodes_buffer, tree_complete, tree_depth, real_depth );

	return node_id;
}

static void buildMeshTree( uint* host_level_split_count,
								bvnode* nodes_buffer, uint& node_count,
											uint* splits_idx, uint tri_count,
													bool tree_complete, uint tree_depth, uint& real_depth )
{
	uint* host_splits_idx;
	
	uint* host_level_splits_idx[32];
	uint* host_level_splits_ptr[32];

	// on établit la liste des indices des séparateurs par niveau
	{
		uint split_count = host_get_split_count();

		cudaMallocHost( (void**)&host_splits_idx, split_count * sizeof(uint) );
		cudaMemcpy( host_splits_idx, splits_idx, split_count * sizeof(uint), cudaMemcpyDeviceToHost );

		host_level_splits_idx[0] = host_splits_idx;
		host_level_splits_ptr[0] = host_splits_idx;

		for ( uint k = 1 ; k <= tree_depth ; k++ )
		{
			host_level_splits_idx[k]
				= host_level_splits_idx[k-1]
				+ host_level_split_count[k-1];

			host_level_splits_ptr[k]
				= host_level_splits_idx[k];
		}
	}

	createNode( host_level_splits_idx,
					host_level_splits_ptr,
							0, tri_count, 0, 
								node_count, 0, nodes_buffer,
									tree_complete, tree_depth, real_depth );

	node_count++;

	real_depth++;

	cudaFreeHost( host_splits_idx );
}

void CUDA_buildMeshTreeLBVH( CUDAMesh* mesh, CUDAMeshTree* tree, bool tree_complete = false, uint tree_depth = 32 )
{
	uint* codes; uint* splits_idx; uint* splits_lvl;
	bvnode* nodes_buffer; bvnode* host_nodes_buffer;

	// on alloue de la mémoire temporaire
	{
		uint count = mesh->getFaceCount();

		cudaMalloc( (void**)&codes, count * sizeof(uint) );
		cudaMalloc( (void**)&splits_idx, count * sizeof(uint) );
		cudaMalloc( (void**)&splits_lvl, count * sizeof(uint) );
		cudaMalloc( (void**)&nodes_buffer, count * sizeof(bvnode) );
		cudaMallocHost( (void**)&host_nodes_buffer, count * sizeof(bvnode) );
	}

	// on calcule les boites englobantes de chacune des primitives
	{
		aabox* boxes = tree->getBoxesArray()->getPointer();
		uint* indices = tree->getIndicesArray()->getPointer();
		uint4* faces = mesh->getFacesArray()->getPointer();
		float4* vertices = mesh->getVerticesArray()->getPointer();
		uint count = mesh->getFaceCount();

		CUDA_calcMeshBoxes( boxes, indices, faces, vertices, count );
	}

	// on calcule le code de Morton pour chaque primitive
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_boxes, tree->getBoxesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_buildMeshTreeLBVH() : Unable to bind CUDA texture." );

		dim3 db = dim3( BLOCK_SIZE );
		dim3 dg = dim3( ( mesh->getFaceCount() + db.x - 1 ) / db.x );

		float4 root_box_min = mesh->getHitbox().min;
		float4 root_box_max = mesh->getHitbox().max;

		uint box_count = mesh->getFaceCount();

		kernel_calcMortonCodes<<<dg,db>>>
		(
			codes,
			root_box_min,
			root_box_max,
			box_count
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_buildMeshTreeLBVH() : Unable to execute CUDA kernel." );

		cudaUnbindTexture( tex_boxes );
	}

	// on trie les primitives par leur code de Morton
	{
		uint* boxes_id = tree->getIndicesArray()->getPointer();
		uint box_count = mesh->getFaceCount();

		// uint bit_depth = fast_log2( box_count );

		CUDA_radixSortGeneral
		(
			codes,
			boxes_id,
			box_count,
			32,	true
		);
	}

	// on construit la liste de paires (id de triangle, niveau de séparation)
	// nb : le niveau de séparation équivaut au poids du bit de morton (fort = 0, faible = 31)
	{
		cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_codes, codes, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_buildMeshTreeLBVH() : Unable to bind CUDA texture." );

		host_set_split_count( 0 );

		uint* p_level_split_count;
		cudaGetSymbolAddress( (void**)&p_level_split_count, "level_split_count" );
		cudaMemset( p_level_split_count, 0, 32 * sizeof(uint) );

		dim3 db = dim3( BLOCK_SIZE );
		dim3 dg = dim3( ( mesh->getFaceCount() + db.x - 1 ) / db.x );

		uint box_count = mesh->getFaceCount();

		kernel_findSplits<<<dg,db>>>
		(
			splits_idx,
			splits_lvl,
			box_count
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_buildMeshTreeLBVH() : Unable to execute CUDA kernel." );

		cudaUnbindTexture( tex_codes );
	}

	uint host_level_split_count[32];

	// on trie désormais la liste des paires selon leur niveau de séparation
	{
		uint split_count = host_get_split_count();

		CUDA_radixSortGeneral
		(
			splits_lvl,
			splits_idx,
			split_count,
			8, false
		);

		cudaMemcpyFromSymbol( host_level_split_count, "level_split_count",
								32 * sizeof(uint), 0, cudaMemcpyDeviceToHost );

		uint* splits_ptr = splits_idx;

		for ( uint k = 1 ; k <= tree_depth ; k++ )
		{
			splits_ptr += host_level_split_count[k-1];

			CUDA_radixSort
			(
				splits_ptr,
				host_level_split_count[k]
			);
		}
	}

	uint host_node_count = 0;

	uint real_depth = 0;

	// on construit l'arbre à partir des paires
	{
		uint tri_count = mesh->getFaceCount();

		buildMeshTree
		(
			host_level_split_count,
			host_nodes_buffer,
			host_node_count,
			splits_idx,
			tri_count,
			tree_complete,
			tree_depth,
			real_depth
		);

		cudaMemcpy( nodes_buffer, host_nodes_buffer, host_node_count * sizeof(bvnode), cudaMemcpyHostToDevice );
	}

	// on calcule les aaboxes des feuilles de l'arbre
	{
		{
			cudaChannelFormatDesc desc = { 32, 0, 0, 0, cudaChannelFormatKindUnsigned };

			if ( cudaBindTexture( NULL, tex_indices, tree->getIndicesArray()->getPointer(), desc ) != cudaSuccess )
				_assert( false, __FILE__, __LINE__, "CUDA_buildMeshTreeLBVH() : Unable to bind CUDA texture." );
		}

		{
			cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

			if ( cudaBindTexture( NULL, tex_boxes, tree->getBoxesArray()->getPointer(), desc ) != cudaSuccess )
				_assert( false, __FILE__, __LINE__, "CUDA_buildMeshTreeLBVH() : Unable to bind CUDA texture." );
		}

		for ( uint k = 0 ; k < real_depth ; k++ )
		{
			dim3 db = dim3( BLOCK_SIZE );
			dim3 dg = dim3( ( host_node_count + db.x - 1 ) / db.x );

			kernel_calcBoundingBoxes<<<dg,db>>>
			(
				nodes_buffer, host_node_count, k
			);

			cudaThreadSynchronize();

			_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
						"CUDA_buildMeshTreeSAH() : Unable to execute CUDA kernel." );
		}

		cudaMemcpy( host_nodes_buffer, nodes_buffer, host_node_count * sizeof(bvnode), cudaMemcpyDeviceToHost );

		tree->updateNodes( nodes_buffer, host_node_count );

		cudaUnbindTexture( tex_boxes );
		cudaUnbindTexture( tex_indices );
	}

	// on libère la mémoire temporaire
	{
		cudaFreeHost( host_nodes_buffer );
		cudaFree( nodes_buffer );
		cudaFree( splits_lvl );
		cudaFree( splits_idx );
		cudaFree( codes );
	}

	// on réordonne si nécessaire les faces du mesh
	if ( tree_complete )
	{
		uint4* faces = mesh->getFacesArray()->getPointer();
		uint* indices = tree->getIndicesArray()->getPointer();
		uint count = mesh->getFaceCount();

		CUDA_reorderMeshFaces( faces, indices, count );
	}

	debugPrint( "vertex_count=%d\n", mesh->getVertexCount() );
	debugPrint( "face_count=%d\n", mesh->getFaceCount() );
	debugPrint( "node_count=%d\n", tree->getNodeCount() );
}
