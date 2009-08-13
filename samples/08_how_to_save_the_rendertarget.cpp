
/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define USE_WIN32
// #define USE_X11

// NB : OpenGL est un poil plus lent que Direct3D mais est la seule API disponible
//      sous Linux et permet par ailleurs de visionner le rendu en mode deviceemu.

#define USE_DIRECT3D
// #define USE_OPENGL

// #define QUICK_MESH_FILE "box2.cdm"
// #define QUICK_MESH_FILE "disk2.cdm"
#define QUICK_MESH_FILE "bigguy.cdm"
// #define QUICK_MESH_FILE "sponza.cdm"
// #define QUICK_MESH_FILE "sibenik.cdm"

#define RENDER_PATH "c:\\render.jpg"

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

void loadScene()
{
	mesh = new CUDAMesh();
	mesh->quickLoad( QUICK_MESH_FILE );

	tree = new CUDAMeshTree( mesh );
	tree->quickLoad( QUICK_MESH_FILE );
}

void unloadScene()
{
	if ( tree )
		delete tree;

	if ( mesh )
		delete mesh;
}

void saveRender()
{
	raytracer->getBuffer()->clearOutputSurfaces();

	raytracer->calcPrimaryRays();

	raytracer->raytraceMeshTree( mesh, tree );

	raytracer->interpolateOutputSurfaces( mesh, true, true, false, false, false );

	raytracer->saveRenderSurface( RENDER_PATH );
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

	loadScene();

	saveRender();

	unloadScene();

	freeRaytracer();

	freeRenderer();

	freeWindow();

	freeEngine();

	return 0;
}
