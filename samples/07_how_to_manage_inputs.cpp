
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
// #define QUICK_MESH_FILE "bigguy.cdm"
#define QUICK_MESH_FILE "sponza.cdm"
// #define QUICK_MESH_FILE "sibenik.cdm"

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

const bool DEBUG_KEY_CODE = false;

const float MOUSE_SENSIVITY = 0.01f;
const float MOVE_SPEED = 50.0f;

CUDAEngine* engine = NULL;
CUDAWindow* window = NULL;
CUDARenderer* renderer = NULL;
CUDARaytracer* raytracer = NULL;
CUDACamera* camera = NULL;
CUDAMesh* mesh = NULL;
CUDAMeshTree* tree = NULL;

bool running;

bool moveForward;
bool moveBackward;
bool strafeLeft;
bool strafeRight;

int lastX, lastY;

/***********************************************************************************/
/** EVENEMENTS                                                                    **/
/***********************************************************************************/

void onClose()
{
	running = false;
}

void onMouseMove( int _keyboardContext, int _mouseContext, int _mouseX, int _mouseY )
{
	if ( raytracer == NULL ) return;

	// si le bouton gauche de la souris est enfoncé alors on pivote la caméra

	if ( _mouseContext & CUDAWindow::MC_LEFT )
	{
		float dx = MOUSE_SENSIVITY * (float)(_mouseX - lastX);
		float dy = MOUSE_SENSIVITY * (float)(_mouseY - lastY);

		camera->rotate( dx, dy, 0.0f );
	}

	lastX = _mouseX;
	lastY = _mouseY;
}

void onKeyDown( int _keyboardContext, int _keyCode )
{
	if ( DEBUG_KEY_CODE )
		printf( "_keyCode = %d\n", _keyCode );

	// on signale lorsqu'une flèche du clavier est enfoncée

	if ( _keyCode == 90 )
		moveForward = true;
	else if ( _keyCode == 83 )
		moveBackward = true;
	else if ( _keyCode == 81 )
		strafeLeft = true;
	else if ( _keyCode == 68 )
		strafeRight = true;

	// on signale lorsque la touche ESC du clavier est enfoncée

	if ( _keyCode == 27 ) running = false;
}

void onKeyUp( int _keyboardContext, int _keyCode )
{
	// on signale lorsqu'une flèche du clavier est relachée

	if ( _keyCode == 90 )
		moveForward = false;
	else if ( _keyCode == 83 )
		moveBackward = false;
	else if ( _keyCode == 81 )
		strafeLeft = false;
	else if ( _keyCode == 68 )
		strafeRight = false;
}

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void mainLoop()
{
	running = true;

	while( running )
	{
		raytracer->getBuffer()->clearOutputSurfaces();

		raytracer->calcPrimaryRays();

		raytracer->raytraceMeshTree( mesh, tree );

		raytracer->interpolateOutputSurfaces( mesh, true, true, false, false, false );

		raytracer->updateRenderSurface();

		renderer->update();

		engine->update();

		// on met à jour la position de la caméra en fonction des touches enfoncées et du temps passé à effectuer la dernière boucle

		if ( moveForward )
			camera->moveForward( MOVE_SPEED * (float)engine->getDelta() );

		if ( moveBackward )
			camera->moveBackward( MOVE_SPEED * (float)engine->getDelta() );

		if ( strafeLeft )
			camera->moveLeft( MOVE_SPEED * (float)engine->getDelta() );

		if ( strafeRight )
			camera->moveRight( MOVE_SPEED * (float)engine->getDelta() );
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

	// enregistrement des callbacks

	window->registerCloseCallback( onClose );
	window->registerKeyDownCallback( onKeyDown );
	window->registerKeyUpCallback( onKeyUp );
	window->registerMouseMoveCallback( onMouseMove );

	window->show();
}

void freeWindow()
{
	window->hide();

	// suppression des callbacks

	window->unregisterCloseCallback( onClose );
	window->unregisterKeyDownCallback( onKeyDown );
	window->unregisterKeyUpCallback( onKeyUp );
	window->unregisterMouseMoveCallback( onMouseMove );

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

	mainLoop();

	unloadScene();

	freeRaytracer();

	freeRenderer();

	freeWindow();

	freeEngine();

	return 0;
}
