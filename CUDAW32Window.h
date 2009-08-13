
#ifdef _WIN32

#ifndef _CUDA_W32WINDOW
#define _CUDA_W32WINDOW

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <windows.h>

#include "CUDAWindow.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDAW32Window : public CUDAWindow
	{

/***********************************************************************************/
/** FONCTIONS AMIES                                                               **/
/***********************************************************************************/

	public:

		friend LRESULT WINAPI windowCallback( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );

		friend DWORD WINAPI windowThread( LPVOID args );

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		WNDCLASSEX wc; // informations sur la fenêtre
		HWND hWnd; // identifiant de la fenêtre

		HANDLE thread;
		HANDLE emptyEvent;
		HANDLE synchronizer;

		bool ready;

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
		CUDAW32Window( CUDAEngine* _engine )
			Instancie la classe et initialise les ressources internes.
		*/
		CUDAW32Window( CUDAEngine* _engine );

		/*
		~CUDAW32Window()
			Libère les ressources internes.
		*/
		virtual ~CUDAW32Window();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getHandle()
			Renvoie le handle associé à cette fenêtre.
		*/
		HWND getHandle() { return hWnd; }

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

	};
}

/***********************************************************************************/

#endif

#endif
