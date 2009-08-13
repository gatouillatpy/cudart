
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAMeshTree.h"
#include "CUDAMesh.h"
#include "CUDAModel.h"

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "CUDADebug.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void CUDA_buildMeshTreeLBVH
(
	renderkit::CUDAMesh* mesh,
	renderkit::CUDAMeshTree* tree,
	bool tree_complete,
	uint tree_depth
);

void CUDA_buildMeshTreeSAH
(
	renderkit::CUDAMesh* mesh,
	renderkit::CUDAMeshTree* tree
);

/***********************************************************************************/

#pragma warning (disable : 4996)

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDAMeshTree::CUDAMeshTree( CUDAMesh* _mesh )
	{
		mesh = _mesh;

		int vertexCount = mesh->getVertexCount();
		int faceCount = mesh->getFaceCount();

		if ( faceCount > 0 )
		{
			boxes = new CUDAArray<aabox>( faceCount );
			indices = new CUDAArray<uint>( faceCount );
		}
		else
		{
			boxes = NULL;
			indices = NULL;
		}

		nodes = NULL;
	}

	CUDAMeshTree::~CUDAMeshTree()
	{
		if ( nodes )
			delete nodes;

		if ( boxes )
			delete boxes;

		if ( indices )
			delete indices;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAMeshTree::buildMeshTreeLBVH( uint tree_depth )
	{
		CUDA_buildMeshTreeLBVH( mesh, this, true, tree_depth );
	}

	void CUDAMeshTree::buildMeshTreeSAH()
	{
		CUDA_buildMeshTreeSAH( mesh, this );
	}

	void CUDAMeshTree::buildMeshTreeHybrid( uint lbvh_tree_depth )
	{
		CUDA_buildMeshTreeLBVH( mesh, this, false, lbvh_tree_depth );

		CUDA_buildMeshTreeSAH( mesh, this );
	}

	void CUDAMeshTree::updateNodes( bvnode* nodes_buffer, uint node_count )
	{
		if ( nodes ) delete nodes;

		nodes = new CUDAArray<bvnode>( node_count );

		nodes->copyDataFromDevice( nodes_buffer );
	}

	void CUDAMeshTree::quickLoad( char* path )
	{
		FILE* file = fopen( path, "rb" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDAMeshTree::quickLoad() : Invalid file." );

		fseek( file, 0, SEEK_END );

		int file_length = ftell( file );

		fseek( file, 0, SEEK_SET );

		const int NODES_HEADER = 0x9077086B;
		const int INDICES_HEADER = 0x2CC8BD7D;
		const int BOXES_HEADER = 0xE54B7842;

		int chunk_header;
		int chunk_offset;
		int chunk_length;

		while ( ( chunk_offset = ftell( file ) ) < file_length )
		{
			fread( &chunk_header, sizeof(int), 1, file );
			fread( &chunk_length, sizeof(int), 1, file );

			if ( chunk_header == NODES_HEADER )
			{
				uint host_size = 0;
				bvnode* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(bvnode) );

				fread( host_data, sizeof(bvnode), host_size, file );

				nodes = new CUDAArray<bvnode>( host_size );
				nodes->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == INDICES_HEADER )
			{
				uint host_size = 0;
				uint* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(uint) );

				fread( host_data, sizeof(uint), host_size, file );

				_assert( indices->getUnitCount() != host_size, __FILE__, __LINE__, "CUDAMeshTree::quickLoad() : Invalid index count." );
				indices->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == BOXES_HEADER )
			{
				uint host_size = 0;
				aabox* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(aabox) );

				fread( host_data, sizeof(aabox), host_size, file );

				_assert( boxes->getUnitCount() != host_size, __FILE__, __LINE__, "CUDAMeshTree::quickLoad() : Invalid box count." );
				boxes->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else
			{
				fseek( file, chunk_offset + chunk_length, SEEK_SET );
			}
		}
	}

	void CUDAMeshTree::quickSave( char* path, bool overwrite )
	{
		FILE* file;

		if ( overwrite )
			file = fopen( path, "wb" );
		else
			file = fopen( path, "ab" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDAMeshTree::quickSave() : Invalid file." );

		fseek( file, 0, SEEK_SET );

		const int NODES_HEADER = 0x9077086B;
		const int INDICES_HEADER = 0x2CC8BD7D;
		const int BOXES_HEADER = 0xE54B7842;

		int chunk_offset;
		int chunk_length;

		#define begin_chunk( header ) \
		{ \
			chunk_offset = ftell( file ); \
			fwrite( &header, sizeof(int), 1, file ); \
			int dummy = -1; \
			fwrite( &dummy, sizeof(int), 1, file ); \
		}

		#define end_chunk() \
		{ \
			chunk_length = ftell( file ) - chunk_offset; \
			fseek( file, chunk_offset + sizeof(int), SEEK_SET ); \
			fwrite( &chunk_length, sizeof(int), 1, file ); \
			fseek( file, chunk_offset + chunk_length, SEEK_SET ); \
		}

		// écriture des noeuds
		if ( nodes )
		{
			begin_chunk( NODES_HEADER );

			bvnode* host_data = NULL;
			uint host_size = nodes->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(bvnode) );

			nodes->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(bvnode), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des indices
		if ( indices )
		{
			begin_chunk( INDICES_HEADER );

			uint* host_data = NULL;
			uint host_size = indices->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(uint) );

			indices->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(uint), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des boites englobantes
		if ( boxes )
		{
			begin_chunk( BOXES_HEADER );

			aabox* host_data = NULL;
			uint host_size = boxes->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(aabox) );

			boxes->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(aabox), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}
	}

}