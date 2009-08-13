
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDAMesh.h"

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

#pragma warning (disable : 4996)

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDAMesh::CUDAMesh()
	{
		model = NULL;

		vertices = NULL;
		normals = NULL;
		colors = NULL;
		texcoords = NULL;

		faces = NULL;
	}

	CUDAMesh::~CUDAMesh()
	{
		if ( faces )
			delete faces;

		if ( texcoords )
			delete texcoords;

		if ( colors )
			delete colors;

		if ( normals )
			delete normals;

		if ( vertices )
			delete vertices;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAMesh::buildFromModel( CUDAModel* _model )
	{
		model = _model;

		hitbox = model->getHitbox();

		if ( model->getVertexCount() > 0 )
		{
			if ( model->hasVertices )
			{
				vertices = new CUDAArray<float4>( model->getVertexCount() );
				vertices->copyDataFromHost( model->getVertexPointer() );
			}
			else
			{
				vertices = NULL;
			}

			if ( model->hasNormals )
			{
				normals = new CUDAArray<float4>( model->getVertexCount() );
				normals->copyDataFromHost( model->getNormalPointer() );
			}
			else
			{
				normals = NULL;
			}

			if ( model->hasColors )
			{
				colors = new CUDAArray<float4>( model->getVertexCount() );
				colors->copyDataFromHost( model->getColorPointer() );
			}
			else
			{
				colors = NULL;
			}

			if ( model->hasTexcoords )
			{
				texcoords = new CUDAArray<float2>( model->getVertexCount() );
				texcoords->copyDataFromHost( model->getTexcoordPointer() );
			}
			else
			{
				texcoords = NULL;
			}
		}
		else
		{
			vertices = NULL;
			normals = NULL;
			colors = NULL;
			texcoords = NULL;
		}

		if ( model->getFaceCount() > 0 )
		{
			faces = new CUDAArray<uint4>( model->getFaceCount() );
			faces->copyDataFromHost( model->getFacePointer() );
		}
		else
		{
			faces = NULL;
		}
	}

	void CUDAMesh::quickLoad( char* path )
	{
		FILE* file = fopen( path, "rb" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDAMesh::quickLoad() : Invalid file." );

		fseek( file, 0, SEEK_END );

		int file_length = ftell( file );

		fseek( file, 0, SEEK_SET );

		const int HITBOX_HEADER = 0x3027AD79;
		const int VERTICES_HEADER = 0xBA5C1CC7;
		const int NORMALS_HEADER = 0x1DD14C48;
		const int COLORS_HEADER = 0x757B31AB;
		const int TEXCOORDS_HEADER = 0x2B77D260;
		const int FACES_HEADER = 0x8E934163;

		int chunk_header;
		int chunk_offset;
		int chunk_length;

		while ( ( chunk_offset = ftell( file ) ) < file_length )
		{
			fread( &chunk_header, sizeof(int), 1, file );
			fread( &chunk_length, sizeof(int), 1, file );

			if ( chunk_header == HITBOX_HEADER )
			{
				fread( &hitbox, sizeof(aabox), 1, file );
			}
			else if ( chunk_header == VERTICES_HEADER )
			{
				uint host_size = 0;
				float4* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

				fread( host_data, sizeof(float4), host_size, file );

				vertices = new CUDAArray<float4>( host_size );
				vertices->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == NORMALS_HEADER )
			{
				uint host_size = 0;
				float4* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

				fread( host_data, sizeof(float4), host_size, file );

				normals = new CUDAArray<float4>( host_size );
				normals->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == COLORS_HEADER )
			{
				uint host_size = 0;
				float4* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

				fread( host_data, sizeof(float4), host_size, file );

				colors = new CUDAArray<float4>( host_size );
				colors->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == TEXCOORDS_HEADER )
			{
				uint host_size = 0;
				float2* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(float2) );

				fread( host_data, sizeof(float2), host_size, file );

				texcoords = new CUDAArray<float2>( host_size );
				texcoords->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else if ( chunk_header == FACES_HEADER )
			{
				uint host_size = 0;
				uint4* host_data = NULL;

				fread( &host_size, sizeof(uint), 1, file );

				cudaMallocHost( (void**)&host_data, host_size * sizeof(uint4) );

				fread( host_data, sizeof(uint4), host_size, file );

				faces = new CUDAArray<uint4>( host_size );
				faces->copyDataFromHost( host_data );

				cudaFreeHost( host_data );
			}
			else
			{
				fseek( file, chunk_offset + chunk_length, SEEK_SET );
			}
		}
	}

	void CUDAMesh::quickSave( char* path, bool overwrite )
	{
		FILE* file;
		
		if ( overwrite )
			file = fopen( path, "wb" );
		else
			file = fopen( path, "ab" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDAMesh::quickSave() : Invalid file." );

		fseek( file, 0, SEEK_SET );

		const int HITBOX_HEADER = 0x3027AD79;
		const int VERTICES_HEADER = 0xBA5C1CC7;
		const int NORMALS_HEADER = 0x1DD14C48;
		const int COLORS_HEADER = 0x757B31AB;
		const int TEXCOORDS_HEADER = 0x2B77D260;
		const int FACES_HEADER = 0x8E934163;

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

		// écriture de la boite englobante
		{
			begin_chunk( HITBOX_HEADER );

			fwrite( &hitbox, sizeof(aabox), 1, file );

			end_chunk();
		}

		// écriture des sommets
		if ( vertices )
		{
			begin_chunk( VERTICES_HEADER );

			float4* host_data = NULL;
			uint host_size = vertices->getUnitCount();
			
			cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

			vertices->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(float4), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des normales
		if ( normals )
		{
			begin_chunk( NORMALS_HEADER );

			float4* host_data = NULL;
			uint host_size = normals->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

			normals->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(float4), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des couleurs
		if ( colors )
		{
			begin_chunk( COLORS_HEADER );

			float4* host_data = NULL;
			uint host_size = colors->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(float4) );

			colors->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(float4), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des coordonnées de texture
		if ( texcoords )
		{
			begin_chunk( TEXCOORDS_HEADER );

			float2* host_data = NULL;
			uint host_size = texcoords->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(float2) );

			texcoords->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(float2), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}

		// écriture des faces
		if ( faces )
		{
			begin_chunk( FACES_HEADER );

			uint4* host_data = NULL;
			uint host_size = faces->getUnitCount();

			cudaMallocHost( (void**)&host_data, host_size * sizeof(uint4) );

			faces->copyDataToHost( host_data );

			fwrite( &host_size, sizeof(uint), 1, file );
			fwrite( host_data, sizeof(uint4), host_size, file );

			cudaFreeHost( host_data );

			end_chunk();
		}
	}

}