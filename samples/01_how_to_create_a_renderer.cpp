
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

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void mainLoop()
{
	while( window->isActive() )
	{
		renderer->update(); // mise à jour du front buffer

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
	// instanciation de la classe

#ifdef USE_DIRECT3D
	renderer = (CUDAD3DRenderer*) new CUDAD3DRenderer( engine );
#endif

#ifdef USE_OPENGL
	renderer = (CUDAOGLRenderer*) new CUDAOGLRenderer( engine );
#endif

	// spécification des propriétés

	renderer->disableFullscreen();
	renderer->disableVSync();
	renderer->useHardware();

	// initialisation du renderer

	renderer->initialize();
}

void freeRenderer()
{
	// arrêt du renderer

	renderer->finalize();

	// suppression de l'instance

	delete renderer;
}

/***********************************************************************************/
/** POINT D'ENTREE                                                                **/
/***********************************************************************************/

int main( int argc, char **argv )
{
	initEngine();

	initWindow();

	initRenderer();

	mainLoop();

	freeRenderer();

	freeWindow();

	freeEngine();

	return 0;
}
