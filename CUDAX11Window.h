
#ifndef _WIN32

#ifndef _CUDA_X11WINDOW
#define _CUDA_X11WINDOW

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDAWindow.h"

#include <GL/glx.h>
#include <X11/extensions/xf86vmode.h>
#include <X11/keysym.h>

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDAX11Window : public CUDAWindow
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		Display* dpy;

		Window win;

		int screen;

		XVisualInfo* vi;

		XF86VidModeModeInfo desk_mode;

		Bool doubleBuffered;

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	protected:

		virtual void create( bool fullscreen = false );
		virtual void destroy();

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAX11Window( CUDAEngine* _engine )
			Instancie la classe et initialise les ressources internes.
		*/
		CUDAX11Window( CUDAEngine* _engine );

		/*
		~CUDAX11Window()
			Libère les ressources internes.
		*/
		virtual ~CUDAX11Window();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getDisplay()
			Renvoie un pointeur vers la structure Display associée à cette fenêtre.
		*/
		Display* getDisplay() { return dpy; }

		/*
		getWindow()
			Renvoie un pointeur vers la structure Window associée à cette fenêtre.
		*/
		Window getWindow() { return win; }

		/*
		getVisualInfo()
			Renvoie un pointeur vers la structure XVisualInfo associée à cette fenêtre.
		*/
		XVisualInfo* getVisualInfo() { return vi; }

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		show()
			Crée la fenêtre si nécessaire puis l'affiche.
		*/
		virtual void show();

		/*
		hide()
			Cache la fenêtre.
		*/
		virtual void hide();

		/*
		updateGeometry()
			Met à jour les dimensions de la fenêtre.
		*/
		void updateGeometry();
		
		/*
		swapBuffers()
			Copie le back buffer dans le frame buffer.
		*/
		void swapBuffers();

	};
}

/***********************************************************************************/

#endif

#else

	#define CUDAX11Window CUDAW32Window

#endif
