
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../../common/Bounds.h"

#include "CUDAModel.h"

#include "Model.h"

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

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDAModel::CUDAModel()
	{
		box.reset();

		hasVertices = false;
		hasNormals = false;
		hasColors = false;
		hasTexcoords = false;

		vertices = NULL;
		normals = NULL;
		colors = NULL;
		texcoords = NULL;
		faces = NULL;

		vertexCount = 0;
		faceCount = 0;
	}

	CUDAModel::~CUDAModel()
	{
		if ( faces )
			free( faces );

		if ( texcoords )
			free( texcoords );

		if ( colors )
			free( colors );

		if ( normals )
			free( normals );

		if ( vertices )
			free( vertices );
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	void CUDAModel::setVertexCount( uint _count )
	{
		if ( hasVertices )
			vertices = (float4*)realloc( vertices, _count * sizeof(float4) );

		if ( hasNormals )
			normals = (float4*)realloc( normals, _count * sizeof(float4) );
	
		if ( hasColors )
			colors = (float4*)realloc( colors, _count * sizeof(float4) );
	
		if ( hasTexcoords )
			texcoords = (float2*)realloc( texcoords, _count * sizeof(float2) );

		vertexCount = _count;
	}

	uint CUDAModel::getVertexCount()
	{
		return vertexCount;
	}

	void CUDAModel::setFaceCount( uint _count )
	{
		faces = (uint4*)realloc( faces, _count * sizeof(uint4) );

		faceCount = _count;
	}

	uint CUDAModel::getFaceCount()
	{
		return faceCount;
	}

	float4* CUDAModel::getVertexPointer()
	{
		return vertices;
	}

	float4* CUDAModel::getNormalPointer()
	{
		return normals;
	}

	float4* CUDAModel::getColorPointer()
	{
		return colors;
	}

	float2* CUDAModel::getTexcoordPointer()
	{
		return texcoords;
	}

	uint4* CUDAModel::getFacePointer()
	{
		return faces;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAModel::insertModel( KModel::Model& model, MAT44 matrix )
	{
		using namespace KModel;

		float4x4 mat_vertex, mat_normal;

		if ( matrix )
		{
			mat_vertex.r0 = make_float4( matrix[0], matrix[4], matrix[ 8], matrix[12] );
			mat_vertex.r1 = make_float4( matrix[1], matrix[5], matrix[ 9], matrix[13] );
			mat_vertex.r2 = make_float4( matrix[2], matrix[6], matrix[10], matrix[14] );
			mat_vertex.r3 = make_float4( matrix[3], matrix[7], matrix[11], matrix[15] );

			matrixTransposeInverse( mat_normal, mat_vertex );
		}

		const int o = getVertexCount();
		const int p = getFaceCount();

		{
			if ( model.hasAttribute( KA_POSITION ) ) hasVertices = true; else hasVertices = false;
			if ( model.hasAttribute( KA_NORMAL ) ) hasNormals = true; else hasNormals = false;
			if ( model.hasAttribute( KA_COLOR ) ) hasColors = true; else hasColors = false;
			if ( model.hasAttribute( KA_TEXCOORD ) ) hasTexcoords = true; else hasTexcoords = false;

			const int n = model.getVertexCount();

			setVertexCount( o + n );

			for ( int k = 0 ; k < n ; k++ )
			{
				Vertex::iterator vertex = model.getVertex(k);

				if ( matrix )
				{
					if ( hasVertices )
					{
						const float* xyz = vertex.getAttribute( KA_POSITION ).getFloatPtr();

						setVertex( o + k, matrixTransform( mat_vertex, make_float4( xyz[0], xyz[1], xyz[2], 1.0f ) ) );
					}

					if ( hasNormals )
					{
						const float* lmn = vertex.getAttribute( KA_NORMAL ).getFloatPtr();

						setNormal( o + k, matrixTransform( mat_normal, make_float4( lmn[0], lmn[1], lmn[2], 1.0f ) ) );
					}
				}
				else
				{
					if ( hasVertices )
						setVertex( o + k, vertex.getAttribute( KA_POSITION ).getFloatPtr() );

					if ( hasNormals )
						setNormal( o + k, vertex.getAttribute( KA_NORMAL ).getFloatPtr() );
				}

				if ( hasColors )
					setColor( o + k, vertex.getAttribute( KA_COLOR ).getFloatPtr() );

				if ( hasTexcoords )
					setTexcoord( o + k, vertex.getAttribute( KA_TEXCOORD ).getFloatPtr() );
			}
		}

		if ( model.hasFlag( MF_IS_TRIANGLES ) )
		{
			const int n = model.getPolygonCount();

			setFaceCount( p + n );

			for ( int k = 0 ; k < n ; k++ )
			{
				Polygon::iterator polygon = model.getPolygon(k);

				const int m = polygon.getVertexCount();
				_assert( m == 3, __FILE__, __LINE__, "CUDAModel::buildFromModel() : Invalid polygon, must be triangle.");

				int a = polygon.getVertex(0).getId();
				int b = polygon.getVertex(1).getId();
				int c = polygon.getVertex(2).getId();

				setFace( p + k, o + a, o + b, o + c, 0 );
			}
		}
	}
	
	void CUDAModel::buildFromModel( KModel::Model& model, MAT44 matrix )
	{
		using namespace KModel;

		float4x4 mat_vertex, mat_normal;

		if ( matrix )
		{
			mat_vertex.r0 = make_float4( matrix[0], matrix[4], matrix[ 8], matrix[12] );
			mat_vertex.r1 = make_float4( matrix[1], matrix[5], matrix[ 9], matrix[13] );
			mat_vertex.r2 = make_float4( matrix[2], matrix[6], matrix[10], matrix[14] );
			mat_vertex.r3 = make_float4( matrix[3], matrix[7], matrix[11], matrix[15] );

			matrixTransposeInverse( mat_normal, mat_vertex );
		}

		{
			if ( model.hasAttribute( KA_POSITION ) ) hasVertices = true; else hasVertices = false;
			if ( model.hasAttribute( KA_NORMAL ) ) hasNormals = true; else hasNormals = false;
			if ( model.hasAttribute( KA_COLOR ) ) hasColors = true; else hasColors = false;
			if ( model.hasAttribute( KA_TEXCOORD ) ) hasTexcoords = true; else hasTexcoords = false;

			const int n = model.getVertexCount();

			setVertexCount( n );

			for ( int k = 0 ; k < n ; k++ )
			{
				Vertex::iterator vertex = model.getVertex(k);

				if ( matrix )
				{
					if ( hasVertices )
					{
						const float* xyz = vertex.getAttribute( KA_POSITION ).getFloatPtr();

						setVertex( k, matrixTransform( mat_vertex, make_float4( xyz[0], xyz[1], xyz[2], 1.0f ) ) );
					}

					if ( hasNormals )
					{
						const float* lmn = vertex.getAttribute( KA_NORMAL ).getFloatPtr();

						setNormal( k, matrixTransform( mat_normal, make_float4( lmn[0], lmn[1], lmn[2], 1.0f ) ) );
					}
				}
				else
				{
					if ( hasVertices )
						setVertex( k, vertex.getAttribute( KA_POSITION ).getFloatPtr() );

					if ( hasNormals )
						setNormal( k, vertex.getAttribute( KA_NORMAL ).getFloatPtr() );
				}

				if ( hasColors )
					setColor( k, vertex.getAttribute( KA_COLOR ).getFloatPtr() );

				if ( hasTexcoords )
					setTexcoord( k, vertex.getAttribute( KA_TEXCOORD ).getFloatPtr() );
			}
		}

		if ( model.hasFlag( MF_IS_TRIANGLES ) )
		{
			const int n = model.getPolygonCount();

			setFaceCount( n );

			for ( int k = 0 ; k < n ; k++ )
			{
				Polygon::iterator polygon = model.getPolygon(k);

				const int m = polygon.getVertexCount();
				_assert( m == 3, __FILE__, __LINE__, "CUDAModel::buildFromModel() : Invalid polygon, must be triangle.");

				int a = polygon.getVertex(0).getId();
				int b = polygon.getVertex(1).getId();
				int c = polygon.getVertex(2).getId();

				setFace( k, a, b, c, 0 );
			}
		}
	}

	void CUDAModel::buildFromModels( KModel::Model* models, MAT44* matrices, int count )
	{
		for ( int k = 0 ; k < count ; k++ )
		{
			insertModel( models[k], matrices[k] );
		}
	}

	void CUDAModel::createCube( float width, float height, float depth )
	{
		hasVertices = true;

		setVertexCount( 8 );

		setVertex( 0, -0.5f * width, -0.5f * height, -0.5f * depth );
		setVertex( 1, -0.5f * width, -0.5f * height, +0.5f * depth );
		setVertex( 2, -0.5f * width, +0.5f * height, -0.5f * depth );
		setVertex( 3, -0.5f * width, +0.5f * height, +0.5f * depth );
		setVertex( 4, +0.5f * width, -0.5f * height, -0.5f * depth );
		setVertex( 5, +0.5f * width, -0.5f * height, +0.5f * depth );
		setVertex( 6, +0.5f * width, +0.5f * height, -0.5f * depth );
		setVertex( 7, +0.5f * width, +0.5f * height, +0.5f * depth );

		setFaceCount( 12 );

		setFace(  0, 0, 1, 2, 0 );
		setFace(  1, 1, 2, 3, 0 );
		setFace(  2, 4, 5, 6, 0 );
		setFace(  3, 5, 6, 7, 0 );

		setFace(  4, 0, 1, 4, 0 );
		setFace(  5, 1, 4, 5, 0 );
		setFace(  6, 2, 3, 6, 0 );
		setFace(  7, 3, 6, 7, 0 );

		setFace(  8, 0, 2, 4, 0 );
		setFace(  9, 2, 4, 6, 0 );
		setFace( 10, 1, 3, 5, 0 );
		setFace( 11, 3, 5, 7, 0 );
	}

}