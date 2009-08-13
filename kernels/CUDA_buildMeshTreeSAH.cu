
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

/***********************************************************************************/
/** TYPES                                                                         **/
/***********************************************************************************/

struct __builtin_align__(16) item
{
	aabox node_box; // boite englobante
	int node_id; // indice de ce noeud

	int obj_id; // indice du premier objet lié à ce noeud
	int obj_bound; // indice du dernier objet + 1

	int padding; // pour conserver l'alignement sur 16 octets
}; // 48 octets

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BLOCK_SIZE 32

#define SM_BIN_COUNT 32

#define STACK_SIZE 4096

#define TRI_RAY_COST 0.125f
#define BOX_RAY_COST 1.000f

#define MIN_LEAF_SIZE 16
#define MAX_LEAF_SIZE 224

/***********************************************************************************/
/** MEMOIRE GLOBALE                                                               **/
/***********************************************************************************/

__device__ int mutex = 0;

__device__ item stack[STACK_SIZE];
__device__ int sp = 0;

__device__ uint node_count;

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <float4, 1, cudaReadModeElementType> tex_boxes;

/***********************************************************************************/
/** MACROS GPU                                                                    **/
/***********************************************************************************/

inline __device__ static void stack_push_global( const int node_id, const aabox& node_box,
														const int obj_id, const int obj_bound )
{
	item i;

	i.node_id = node_id;
	i.node_box = node_box;

	i.obj_id = obj_id;
	i.obj_bound = obj_bound;

	int t_sp = atomicAdd( &sp, 1 );

	stack[t_sp] = i;
}

inline __device__ static void stack_push( const int node_id, const aabox& node_box,
												const int obj_id, const int obj_bound )
{
	// le premier thread du bloc se charge des opérations d'accès à la pile ;
	// on vérouille l'accès de la pile aux autres blocs
	if ( threadIdx.x == 0 )
	{
		__shared__ item i;

		i.node_id = node_id;
		i.node_box = node_box;

		i.obj_id = obj_id;
		i.obj_bound = obj_bound;

		int t_sp = atomicAdd( &sp, 1 );

		stack[t_sp] = i;
	}

	// les autres threads attendent que le premier ait terminé
	__syncthreads();
}

inline __device__ static item stack_pop()
{
	__shared__ item i;

	// le premier thread du bloc se charge des opérations d'accès à la pile ;
	// on vérouille l'accès de la pile aux autres blocs
	if ( threadIdx.x == 0 )
	{
		int t_sp = atomicSub( &sp, 1 ) - 1;

		i = stack[t_sp];
	}

	// les autres threads attendent que le premier ait terminé
	__syncthreads();

	return i;
}

inline __device__ static void make_node( bvnode& bvn, const aabox& box, const int axis, const int child_id )
{
	// on a plus de chances d'avoir un peu de mémoire partagée restante
	// que des registres, or s'il n'y a plus suffisament de registres
	// alors ptx risque de mettre cela dans la mémoire locale, ce qui
	// risque d'être violent pour les performances...

	bvn.node.type = 0;
	bvn.node.axis = axis;
	bvn.node.child_id = child_id;
	bvn.box_min = make_float3( box.min );
	bvn.box_max = make_float3( box.max );
}

inline __device__ static void make_leaf( bvnode& bvn, const aabox& box, const int obj_id, const int obj_count )
{
	bvn.leaf.type = 1;
	bvn.leaf.obj_count = obj_count;
	bvn.leaf.obj_id = obj_id;
	bvn.box_min = make_float3( box.min );
	bvn.box_max = make_float3( box.max );
}

/***********************************************************************************/
/** MACROS CPU                                                                    **/
/***********************************************************************************/

inline __host__ static void host_sp_set( int _sp )
{
	cudaMemcpyToSymbol( "sp", &_sp, sizeof(int), 0, cudaMemcpyHostToDevice );
}

inline __host__ static int host_sp_get()
{
	int _sp; cudaMemcpyFromSymbol( &_sp, "sp", sizeof(int), 0, cudaMemcpyDeviceToHost ); return _sp;
}

inline __host__ void host_stack_push( const int node_id, const aabox& node_box,
													const int obj_id, const int obj_bound )
{
	item i;

	i.node_id = node_id;
	i.node_box = node_box;

	i.obj_id = obj_id;
	i.obj_bound = obj_bound;

	int sp = host_sp_get();

	cudaMemcpyToSymbol( "stack", &i, sizeof(item), (sp++) * sizeof(item), cudaMemcpyHostToDevice );

	host_sp_set(sp);
}

inline __host__ item host_stack_pop()
{
	item i;

	int sp = host_sp_get();

	cudaMemcpyFromSymbol( &i, "stack", sizeof(item), (--sp) * sizeof(item), cudaMemcpyDeviceToHost );

	host_sp_set(sp);

	return i;
}

inline __host__ static void host_set_node_count( uint _node_count )
{
	cudaMemcpyToSymbol( "node_count", &_node_count, sizeof(uint), 0, cudaMemcpyHostToDevice );
}

inline __host__ static uint host_get_node_count()
{
	uint _node_count; cudaMemcpyFromSymbol( &_node_count, "node_count", sizeof(uint), 0, cudaMemcpyDeviceToHost ); return _node_count;
}

/***********************************************************************************/
/** NOYAU                                                                         **/
/***********************************************************************************/

// idée générale :
//   - chaque bloc s'occupe d'un élément de la pile
//   - chaque thread s'occupe si nécessaire d'un objet pour le noeud en cours de traitement

__global__ static void kernel_buildMeshTree( uint* indices, uint* indices_buffer, bvnode* nodes_buffer )
{
	// on ne tient pas compte de threadIdx.y
	// car sa valeur est toujours nulle
	const int& n = threadIdx.x;

	const item& i = stack_pop();

	// étrangement si on met __shared__ ici, le noyau s'éxécute plus
	// lentement ; or si on regarde le fichier ptx, on constate que sans
	// __shared__ t_node est stocké en mémoire locale
	bvnode t_node;

	// s'il n'y a plus que n objets ou moins dans le
	// noeud alors le premier thread en fait une feuille
	if ( i.obj_bound - i.obj_id <= MIN_LEAF_SIZE )
	{
		if ( n == 0 )
		{
			make_leaf
			(
				t_node,
				i.node_box,
				i.obj_id,
				i.obj_bound - i.obj_id
			);

			nodes_buffer[i.node_id] = t_node;
		}

		return;
	}

	volatile int p = i.obj_id - (i.obj_id % BLOCK_SIZE);
	volatile int q = i.obj_bound - (i.obj_bound % BLOCK_SIZE);

	__shared__ aabox s_boxes[BLOCK_SIZE];

	__shared__ int s_axis;

	__shared__ float s_bounds_scale;
	__shared__ float s_bounds_min;

	__shared__ aabox t_box[BLOCK_SIZE];

	float4 t_center;

	int k, l, m, o, a, b;

	int t_id;

	// on calcule l'aabox du centre des objets a inserer
	{
		s_boxes[n].reset();

		// d'abord chaque thread calcule sa aabox avec un certain
		// nombre de primitives...
		for ( k = p ; k <= q ; k += BLOCK_SIZE )
		{
			l = k + n;

			if ( l >= i.obj_id && l < i.obj_bound )
			{
				t_id = indices[l];

				t_box[n].min = tex1Dfetch( tex_boxes, 2 * t_id + 0 );
				t_box[n].max = tex1Dfetch( tex_boxes, 2 * t_id + 1 );

				t_center = 0.5f * ( t_box[n].min + t_box[n].max );

				s_boxes[n].merge( t_center );
			}
		}

		__syncthreads();

		// ...puis le premier thread calcule l'aabox globale
		// de ce noeud et determine son axe le plus long
		if ( n == 0 )
		{
			// #pragma unroll 1
			for ( k = 1 ; k < BLOCK_SIZE ; k++ )
			{
				s_boxes[0].merge( s_boxes[k] );
			}

			t_center = s_boxes[0].max - s_boxes[0].min;

			if ( t_center.x > t_center.y )
			{
				if ( t_center.x > t_center.z )
					s_axis = 0;
				else
					s_axis = 2;
			}
			else
			{
				if ( t_center.y > t_center.z )
					s_axis = 1;
				else
					s_axis = 2;
			}

			if ( s_axis == 0 )
			{
				s_bounds_scale = (float)(SM_BIN_COUNT - 1) / t_center.x;
				s_bounds_min = s_boxes[0].min.x;
			}
			else if ( s_axis == 1 )
			{
				s_bounds_scale = (float)(SM_BIN_COUNT - 1) / t_center.y;
				s_bounds_min = s_boxes[0].min.y;
			}
			else
			{
				s_bounds_scale = (float)(SM_BIN_COUNT - 1) / t_center.z;
				s_bounds_min = s_boxes[0].min.z;
			}
		}

		__syncthreads();
	}

	volatile int axis = s_axis;

	volatile float bounds_scale = s_bounds_scale;
	volatile float bounds_min = s_bounds_min;

	__shared__ bvbin s_bins[SM_BIN_COUNT];

	__shared__ uint s_bins_id[BLOCK_SIZE];

	// on initialise les bins puis on y insere les objets
	{
		// chaque thread initialise ses bins
		for ( m = n ; m < SM_BIN_COUNT ; m += BLOCK_SIZE )
			s_bins[m].reset();

		for ( k = p ; k <= q ; k += BLOCK_SIZE )
		{
			l = k + n;

			if ( l >= i.obj_id && l < i.obj_bound )
			{
				t_id = indices[l];

				s_boxes[n].min = tex1Dfetch( tex_boxes, 2*t_id+0 );
				s_boxes[n].max = tex1Dfetch( tex_boxes, 2*t_id+1 );

				t_center = 0.5f * (s_boxes[n].min + s_boxes[n].max);

				if ( axis == 0 )
					m = (int)( ( t_center.x - bounds_min ) * bounds_scale );
				else if ( axis == 1 )
					m = (int)( ( t_center.y - bounds_min ) * bounds_scale );
				else
					m = (int)( ( t_center.z - bounds_min ) * bounds_scale );

				if ( m < 0 )
					m = 0;
				else if ( m >= SM_BIN_COUNT - 1 )
					m = SM_BIN_COUNT - 1;

				s_bins_id[n] = m;
			}

			__syncthreads();

			if ( n == 0 )
			{
				a = ( k == p ) ? (i.obj_id - p) : 0;
				b = ( k == q ) ? (i.obj_bound - q) : BLOCK_SIZE;

				for ( o = a ; o < b ; o++ )
				{
					aabox& box = s_boxes[o];

					m = s_bins_id[o];

					s_bins[m].insertObject( box );
				}
			}

			__syncthreads();
		}
	}

	__shared__ float T_min;
	__shared__ int m_min;

	__shared__ aabox left_box;
	__shared__ aabox right_box;

	__shared__ int left_offset;
	__shared__ int right_offset;

	__shared__ int left_count;
	__shared__ int right_count;

	// le premier thread recherche le meilleur plan de coupe entre les bins
	if ( n == 0 )
	{
		// on parcourt les bins de min a max ...
		{
			left_box.reset();
			left_count = 0;

			for ( m = 0 ; m < SM_BIN_COUNT - 1 ; m++ )
			{
				bvbin& bin = s_bins[m];

				left_count += bin.obj_count;
				left_box.merge( bin.box_min, bin.box_max );

				// les candidats se trouvent "entre" les bins, le premier a
				// droite de s_bins[0] et le dernier a gauche de s_bins[MAX-1]
				s_bins[m].insertLeft( left_count, left_box );
			}
		}

		T_min = +CUDART_NORM_HUGE_F;
		m_min = -1;

		// ... puis de max a min et on calcule le cout associé aux candidats
		{
			right_box.reset();
			right_count = 0;

			for ( m = SM_BIN_COUNT - 1 ; m > 0 ; m-- )
			{
				bvbin& bin = s_bins[m];

				right_count += bin.obj_count;
				right_box.merge( bin.box_min, bin.box_max );
		        
				// on stocke le resultat dans le bin associe au candidat
				// rappel : les candidats sont a droite de chaque bin
				s_bins[m-1].insertRight( right_count, right_box );
		        
				// on calcule le cout du candidat...
				const float T = s_bins[m-1].getSAH( i.node_box, BOX_RAY_COST, TRI_RAY_COST );

				// ...et on conserve le meilleur
				if ( T < T_min )
				{
					T_min = T;
					m_min = m - 1; // coupe a droite de bin[k-1] et a gauche de bin[k]
				}
			}
		}
	}

	__syncthreads();

	// on termine la construction, si possible
	if ( ( i.obj_bound - i.obj_id < MAX_LEAF_SIZE )
	  && ( T_min > (float)(i.obj_bound - i.obj_id + 1) * TRI_RAY_COST ) )
	{
		if ( n == 0 )
		{
			make_leaf
			(
				t_node,
				i.node_box,
				i.obj_id,
				i.obj_bound - i.obj_id
			);

			nodes_buffer[i.node_id] = t_node;
		}

		return;
	}

	// on reconstruit la meilleure repartition pour T = T_min
	{
		// le premier thread remet à jour les aaboxes des deux fils
		if ( n == 0 )
		{
			left_count = 0;
			right_count = 0;

			left_box.reset();
			right_box.reset();

			for ( m = 0 ; m <= m_min ; m++ )
			{
				bvbin& bin = s_bins[m];

				left_count += bin.obj_count;
				left_box.merge( bin.box_min, bin.box_max );
			}

			for ( m = m_min + 1 ; m < SM_BIN_COUNT ; m++ )
			{
				bvbin& bin = s_bins[m];

				right_count += bin.obj_count;
				right_box.merge( bin.box_min, bin.box_max );
			}

			left_offset = i.obj_id;
			right_offset = i.obj_id + left_count;

			left_count = 0;
			right_count = 0;
		}

		__syncthreads();

		__shared__ int s_indices[BLOCK_SIZE];

		for ( k = p ; k <= q ; k += BLOCK_SIZE )
		{
			l = k + n;

			// chaque thread recherche les bins associés
			// à un certain nombre de aaboxes ...
			if ( l >= i.obj_id && l < i.obj_bound )
			{
				t_id = indices[l];

				s_boxes[n].min = tex1Dfetch( tex_boxes, 2 * t_id + 0 );
				s_boxes[n].max = tex1Dfetch( tex_boxes, 2 * t_id + 1 );

				t_center = 0.5f * (s_boxes[n].min + s_boxes[n].max);

				if ( axis == 0 )
					m = (int)( ( t_center.x - bounds_min ) * bounds_scale );
				else if ( axis == 1 )
					m = (int)( ( t_center.y - bounds_min ) * bounds_scale );
				else
					m = (int)( ( t_center.z - bounds_min ) * bounds_scale );

				if ( m < 0 )
					m = 0;
				else if ( m >= SM_BIN_COUNT - 1 )
					m = SM_BIN_COUNT - 1;

				s_bins_id[n] = m;
				s_indices[n] = t_id;
			}

			__syncthreads();

			// ... puis le premier thread du bloc identifie le
			// nouvel id de chaque triangle en fonction du bin
			// dans lequel sa aabox se trouve
			if ( n == 0 )
			{
				a = ( k == p ) ? (i.obj_id - p) : 0;
				b = ( k == q ) ? (i.obj_bound - q) : BLOCK_SIZE;

				for ( o = a ; o < b ; o++ )
				{
					m = s_bins_id[o];

			        if ( m <= m_min )
					{
						// inserer l'objet dans le fils gauche
						indices_buffer[left_offset+left_count] = s_indices[o];
						left_count++;
					}
					else
					{
						// inserer l'objet dans le fils droit
						indices_buffer[right_offset+right_count] = s_indices[o];
						right_count++;
					}
				}
			}

			__syncthreads();
		}

		// enfin chaque thread recopie son lot de valeurs depuis indices_buffer
		// vers indices ; ça semble galère pour une simple copie mais cela permet
		// juste que celle-ci soit faite de manière "coalescente" (voir doc CUDA)
		{
			for ( k = p ; k <= q ; k += BLOCK_SIZE )
			{
				l = k + n;

				if ( l >= i.obj_id && l < i.obj_bound )
					indices[l] = indices_buffer[l];
			}
		}

		__syncthreads();
	}

	__shared__ int left_child_id;
	__shared__ int right_child_id;

	// le premier thread incrémente de 2 le nombre de noeuds de manière
	// atomique au cas où un autre bloc y accède aussi ; puis définit l'id
	// des deux nouveaux noeuds enfants ; et enfin stocke le noeud actuel
	if ( n == 0 )
	{
		// count = node_count;
		// node_count += 2;
		uint count = atomicAdd( &node_count, 2 );

		left_child_id = count + 0;
		right_child_id = count + 1;

		make_node
		(
			t_node,
			i.node_box,
			axis,
			count
		);

		nodes_buffer[i.node_id] = t_node;
	}

	__syncthreads();

	stack_push( right_child_id, right_box, right_offset, right_offset + right_count );
	stack_push( left_child_id, left_box, left_offset, left_offset + left_count );
}

__global__ static void kernel_initStack( bvnode* nodes_buffer, uint node_count )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < node_count )
	{
		bvnode bvn = nodes_buffer[n];

		if ( isLeaf( bvn ) )
		{
			aabox node_box;
			node_box.reset( bvn.box_min, bvn.box_max );

			uint tri_index = bvn.leaf.obj_id;
			uint tri_bound = bvn.leaf.obj_id + bvn.leaf.obj_count;

			stack_push_global( n, node_box, tri_index, tri_bound );
		}
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

void CUDA_buildMeshTreeSAH( CUDAMesh* mesh, CUDAMeshTree* tree )
{
	uint* indices_buffer; bvnode* nodes_buffer;

	// on alloue de la mémoire temporaire
	{
		cudaMalloc( (void**)&indices_buffer, mesh->getFaceCount() * sizeof(uint) );
		cudaMalloc( (void**)&nodes_buffer, mesh->getFaceCount() * sizeof(bvnode) );
	}

	// on calcule si nécessaire les boites englobantes de chacune des primitives
	if ( tree->getNodeCount() == 0 )
	{
		aabox* boxes = tree->getBoxesArray()->getPointer();
		uint* indices = tree->getIndicesArray()->getPointer();
		uint4* faces = mesh->getFacesArray()->getPointer();
		float4* vertices = mesh->getVerticesArray()->getPointer();
		uint count = mesh->getFaceCount();

		CUDA_calcMeshBoxes( boxes, indices, faces, vertices, count );
	}

	// on monte les textures pour les données en entrée
	{
		cudaChannelFormatDesc desc = { 32, 32, 32, 32, cudaChannelFormatKindFloat };

		if ( cudaBindTexture( NULL, tex_boxes, tree->getBoxesArray()->getPointer(), desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_buildMeshTreeSAH() : Unable to bind CUDA texture." );
	}

	// on initialise la construction
	if ( tree->getNodeCount() == 0 )
	{
		host_sp_set( 0 );

		host_stack_push( 0, mesh->getHitbox(), 0, mesh->getFaceCount() );

		host_set_node_count( 1 );
	}
	else
	{
		uint host_node_count = tree->getNodeCount();
		bvnode* node_pointer = tree->getNodesArray()->getPointer();

		host_set_node_count( host_node_count );

		cudaMemcpy( nodes_buffer, node_pointer, host_node_count * sizeof(bvnode), cudaMemcpyDeviceToDevice );

		host_sp_set( 0 );

		dim3 db = dim3( BLOCK_SIZE );
		dim3 dg = dim3( ( host_node_count + db.x - 1 ) / db.x );

		kernel_initStack<<<dg,db>>>
		(
			nodes_buffer,
			host_node_count
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_buildMeshTreeSAH() : Unable to execute CUDA kernel." );
	}

	// on construit l'arbre
	{
		int sp;
	 
		while ( ( sp = host_sp_get() ) > 0 )
		{
			_assert( sp < STACK_SIZE, __FILE__, __LINE__, "CUDA_buildMeshTreeSAH() : Stack overflow." );

			dim3 db = dim3( BLOCK_SIZE );
			dim3 dg = dim3( sp );

			uint* indices = tree->getIndicesArray()->getPointer();

			kernel_buildMeshTree<<<dg,db>>>
			(
				indices,
				indices_buffer,
				nodes_buffer
			);

			cudaThreadSynchronize();

			_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
						"CUDA_buildMeshTreeSAH() : Unable to execute CUDA kernel." );
		}

		tree->updateNodes( nodes_buffer, host_get_node_count() );
	}

	// on libère la mémoire temporaire
	{
		cudaFree( nodes_buffer );
		cudaFree( indices_buffer );
	}

	// on démonte les textures
	{
		cudaUnbindTexture( tex_boxes );
	}

	// on réordonne les faces du mesh
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
