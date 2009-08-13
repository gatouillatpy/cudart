
/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define USE_WIN32
// #define USE_X11

// NB : OpenGL est un poil plus lent que Direct3D mais est la seule API disponible
//      sous Linux et permet par ailleurs de visionner le rendu en mode deviceemu.

#define USE_DIRECT3D
// #define USE_OPENGL

// #define IMPORT_MODEL_FILE "box2.obj"
// #define IMPORT_MODEL_FILE "disk2.obj"
// #define IMPORT_MODEL_FILE "bigguy.obj"
#define IMPORT_MODEL_FILE "sponza.obj"
// #define IMPORT_MODEL_FILE "sibenik.obj"

// #define QUICK_MESH_FILE "box2.cdm"
// #define QUICK_MESH_FILE "disk2.cdm"
// #define QUICK_MESH_FILE "bigguy.cdm"
#define QUICK_MESH_FILE "sponza.cdm"
// #define QUICK_MESH_FILE "sibenik.cdm"

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../../../common/Bounds.h"

#include "../CUDACommon.h"

#include "../CUDAModel.h"
#include "../CUDAMesh.h"
#include "../CUDAMeshTree.h"

#include "MemoryManager.h"
#include "MaterialManager.h"
#include "Model.h"
#include "ModelMaya.h"
#include "ModelGeometry.h"
#include "ModelClean.h"

#pragma warning (disable : 4996)

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "../CUDADebug.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/***********************************************************************************/

using namespace renderkit;
using namespace KModel;

/***********************************************************************************/
/** VARIABLES GLOBALES                                                            **/
/***********************************************************************************/

CUDAModel* model = NULL;
CUDAMesh* mesh = NULL;
CUDAMeshTree* tree = NULL;

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void createScene()
{
	MemoryManager::initAllPools( INT_MAX );

	ResourceFactory managers;
	MaterialManager materials( managers );

	Model original_model, clean_model;

	{
		assert( LoadMayaFromFile( original_model, materials, IMPORT_MODEL_FILE ) >= 0 );
		TriangulateInPlace( original_model );

		ModelCopyClean( clean_model, original_model );
		BuildVertexNormals( clean_model );

		MAT44 scale_matrix = { 2.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 2.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 2.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f };

		model = new CUDAModel();
		model->buildFromModel( clean_model, scale_matrix );
	}

	{
		mesh = new CUDAMesh();
		mesh->buildFromModel( model );

		cudaEvent_t e_start, e_stop;

		cudaEventCreate( &e_start );
		cudaEventCreate( &e_stop );

		tree = new CUDAMeshTree( mesh );

		cudaEventRecord( e_start, 0 );

		tree->buildMeshTreeSAH();

		cudaEventRecord( e_stop, 0 );
		cudaEventSynchronize( e_stop );

		float time; cudaEventElapsedTime( &time, e_start, e_stop );

		cudaEventDestroy( e_stop );
		cudaEventDestroy( e_start );

		debugPrint( ">> construction de l'arbre en %.3f ms\n", time );
	}
}

void quickSaveScene()
{
	mesh->quickSave( QUICK_MESH_FILE );

	tree->quickSave( QUICK_MESH_FILE );
}

void destroyScene()
{
	// suppression des instances

	if ( tree )
		delete tree;

	if ( mesh )
		delete mesh;

	if ( model )
		delete model;
}

/***********************************************************************************/
/** POINT D'ENTREE                                                                **/
/***********************************************************************************/

int main( int argc, char **argv )
{
	createScene();

	quickSaveScene();

	destroyScene();

	return 0;
}
