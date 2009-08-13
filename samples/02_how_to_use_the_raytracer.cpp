
/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define USE_WIN32
// #define USE_X11

// NB : OpenGL est un poil plus lent que Direct3D mais est la seule API disponible
//      sous Linux et permet par ailleurs de visionner le rendu en mode deviceemu.

#define USE_DIRECT3D
// #define USE_OPENGL

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

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
		raytracer->getBuffer()->clearOutputSurfaces(); // remise à zéro de la surface de rendu

		raytracer->calcPrimaryRays(); // calcul des rayons primaires à partir des données de la caméra

		raytracer->raytraceMeshTree( mesh, tree ); // raytracing du mesh

		raytracer->interpolateOutputSurfaces( mesh, true, true, false, false, false ); // interpolation des points et normales pour les pixels dont le rayon intersecte une face

		raytracer->updateRenderSurface(); // mise à jour du back buffer

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
	// instanciation de la classe

	raytracer = new CUDARaytracer( engine );

	// spécification des propriétés de la caméra

	camera = raytracer->getCamera();

	camera->setRatio( renderer->getSurface()->getWidth(),
		renderer->getSurface()->getHeight() );
	camera->setCenter( -10.0f, 20.0f, 30.0f );
	camera->lookAt( 0.0f, 0.0f, 0.0f );
	camera->update();

	// initialisation du raytracer

	raytracer->initialize();
}

void freeRaytracer()
{
	// arrêt du raytracer

	raytracer->finalize();

	// suppression de l'instance
	
	delete raytracer;
}

void createScene()
{
	// création d'un modèle de cube

	model = new CUDAModel();
	model->createCube( 20.0f, 20.0f, 20.0f );

	// construction en mémoire vidéo du mesh à partir de ce modèle

	mesh = new CUDAMesh();
	mesh->buildFromModel( model );

	// construction de l'arbre binaire pour ce mesh

	tree = new CUDAMeshTree( mesh );
	tree->buildMeshTreeSAH();
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
