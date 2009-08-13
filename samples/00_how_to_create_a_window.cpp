
/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define USE_WIN32
// #define USE_X11

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

#include "../CUDAEngine.h"
#include "../CUDAW32Window.h"
#include "../CUDAX11Window.h"

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

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void mainLoop()
{
	while( window->isActive() )
	{
		// calcul du temps écoulé entre deux boucles

		engine->update();
	}
}

void initWindow()
{
	// instanciation de la classe

#ifdef USE_WIN32
	window = (CUDAW32Window*) new CUDAW32Window( engine );
#endif

#ifdef USE_X11
	window = (CUDAX11Window*) new CUDAX11Window( engine );
#endif

	// spécification des propriétés

	window->setLeft( 120 );
	window->setTop( 80 );
	window->setWidth( 400 );
	window->setHeight( 300 );

	// affichage de la fenêtre

	window->show();
}

void freeWindow()
{
	// masquage de la fenêtre

	window->hide();

	// suppression de l'instance

	delete window;
}

void initEngine()
{
	// instanciation de la classe

	engine = new CUDAEngine();
}

void freeEngine()
{
	// suppression de l'instance

	delete engine;
}

/***********************************************************************************/
/** POINT D'ENTREE                                                                **/
/***********************************************************************************/

int main( int argc, char **argv )
{
	initEngine();

	initWindow();

	mainLoop();

	freeWindow();

	freeEngine();

	return 0;
}
