
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
#define IMPORT_MODEL_FILE "bigguy.obj"
// #define IMPORT_MODEL_FILE "sponza.obj"
// #define IMPORT_MODEL_FILE "sibenik.obj"

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../../../common/Bounds.h"

#include "../CUDACommon.h"

#include "../CUDAEngine.h"
#include "../CUDAW32Window.h"
#include "../CUDAX11Window.h"
#include "../CUDAD3DRenderer.h"
#include "../CUDAOGLRenderer.h"
#include "../CUDARaytracer.h"
#include "../CUDACamera.h"
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

CUDAEngine* engine = NULL;
CUDAWindow* window = NULL;
CUDARenderer* renderer = NULL;
CUDARaytracer* raytracer = NULL;
CUDACamera* camera = NULL;
CUDAModel* model = NULL;
CUDAMesh* mesh = NULL;
CUDAMeshTree* tree = NULL;

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void mainLoop()
{
	while( window->isActive() )
	{
		raytracer->getBuffer()->clearOutputSurfaces();

		raytracer->calcPrimaryRays();

		raytracer->raytraceMeshTree( mesh, tree );

		raytracer->interpolateOutputSurfaces( mesh, true, true, false, false, false );

		raytracer->updateRenderSurface();

		renderer->update();

		engine->update();
	}
}

void initWindow()
{
#ifdef USE_WIN32
	window = (CUDAW32Window*) new CUDAW32Window( engine );
#endif

#ifdef USE_X11
	window = (CUDAX11Window*) new CUDAX11Window( engine );
#endif

	window->setLeft( 120 );
	window->setTop( 80 );
	window->setWidth( 400 );
	window->setHeight( 300 );

	window->show();
}

void freeWindow()
{
	window->hide();

	delete window;
}

void initEngine()
{
	engine = new CUDAEngine();
}

void freeEngine()
{
	delete engine;
}

void initRenderer()
{
#ifdef USE_DIRECT3D
	renderer = (CUDAD3DRenderer*) new CUDAD3DRenderer( engine );
#endif

#ifdef USE_OPENGL
	renderer = (CUDAOGLRenderer*) new CUDAOGLRenderer( engine );
#endif

	renderer->disableFullscreen();
	renderer->disableVSync();
	renderer->useHardware();

	renderer->initialize();
}

void freeRenderer()
{
	renderer->finalize();

	delete renderer;
}

void initRaytracer()
{
	raytracer = new CUDARaytracer( engine );

	camera = raytracer->getCamera();

	camera->setRatio( renderer->getSurface()->getWidth(),
		renderer->getSurface()->getHeight() );
	camera->setCenter( -10.0f, 20.0f, 30.0f );
	camera->lookAt( 0.0f, 0.0f, 0.0f );
	camera->update();

	raytracer->initialize();
}

void freeRaytracer()
{
	raytracer->finalize();

	delete raytracer;
}

void createScene()
{
	MemoryManager::initAllPools( INT_MAX );

	ResourceFactory managers;
	MaterialManager materials( managers );

	Model original_model, clean_model;

	// chargement du modèle 3d
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

	// construction du mesh et de son arbre binaire en mémoire vidéo à partir du modèle 3d
	{
		mesh = new CUDAMesh();
		mesh->buildFromModel( model );

		cudaEvent_t e_start, e_stop;

		cudaEventCreate( &e_start );
		cudaEventCreate( &e_stop );

		tree = new CUDAMeshTree( mesh );

		cudaEventRecord( e_start, 0 );

		tree->buildMeshTreeSAH();			// construction d'un arbre binaire avec une heuristique SAH (meilleure qualité mais temps de construction relativement élevé)
		//tree->buildMeshTreeLBVH( 2 );		// construction d'un arbre binaire sur deux niveaux, c'est à dire équivalent d'un quadtree (temps de construction optimal mais qualité médiocre)
		//tree->buildMeshTreeLBVH( 3 );		// idem sur trois niveaux, c'est à dire équivalent d'un octree
		//tree->buildMeshTreeLBVH( 6 );		// idem sur six niveaux, c'est à dire équivalent d'un octree à deux niveaux
		//tree->buildMeshTreeLBVH( 12 );	// idem sur douze niveaux, c'est à dire équivalent d'un octree à quatre niveaux
		//tree->buildMeshTreeHybrid( 3 );	// construction d'un arbre binaire de manière hybride, c'est à dire en construisant un octree sur la racine puis en finissant avec une heuristique SAH (bon compromis entre qualité et temps de construction)

		cudaEventRecord( e_stop, 0 );
		cudaEventSynchronize( e_stop );

		float time; cudaEventElapsedTime( &time, e_start, e_stop );

		cudaEventDestroy( e_stop );
		cudaEventDestroy( e_start );

		debugPrint( ">> construction de l'arbre en %.3f ms\n", time );
	}
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
	initEngine();

	initWindow();

	initRenderer();

	initRaytracer();

	createScene();

	mainLoop();

	destroyScene();

	freeRaytracer();

	freeRenderer();

	freeWindow();

	freeEngine();

	return 0;
}
