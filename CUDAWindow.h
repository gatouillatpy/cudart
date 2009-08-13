
#ifndef _CUDA_WINDOW
#define _CUDA_WINDOW

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <vector>

using namespace std;

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDAWindow
	{

/***********************************************************************************/
/** TYPES                                                                         **/
/***********************************************************************************/

	public:

		typedef void ( *LPCLOSECALLBACK ) ( );

		typedef void ( *LPKEYUPCALLBACK ) ( int _keyboardContext, int _keyCode );
		typedef void ( *LPKEYDOWNCALLBACK ) ( int _keyboardContext, int _keyCode );

		typedef void ( *LPMOUSEMOVECALLBACK ) ( int _keyboardContext, int _mouseContext, int _mouseX, int _mouseY ) ;
		typedef void ( *LPMOUSEUPCALLBACK ) ( int _keyboardContext, int _mouseContext, int _mouseX, int _mouseY ) ;
		typedef void ( *LPMOUSEDOWNCALLBACK ) ( int _keyboardContext, int _mouseContext, int _mouseX, int _mouseY ) ;

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		CUDAEngine* engine;

		vector<LPCLOSECALLBACK> closeCallbacks;

		vector<LPKEYUPCALLBACK> keyUpCallbacks;
		vector<LPKEYDOWNCALLBACK> keyDownCallbacks;

		vector<LPMOUSEMOVECALLBACK> mouseMoveCallbacks;
		vector<LPMOUSEUPCALLBACK> mouseUpCallbacks;
		vector<LPMOUSEDOWNCALLBACK> mouseDownCallbacks;

		int keyboardContext;
		int mouseContext;

		int mouseX;
		int mouseY;

		bool visible;
		bool active;

		int left;
		int top;
		int width;
		int height;

		int innerWidth;
		int innerHeight;

		bool fullscreen;

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	protected:

		virtual void create( bool fullscreen ) = 0;
		virtual void destroy() = 0;

/***********************************************************************************/
/** EVENEMENTS                                                                    **/
/***********************************************************************************/

	protected:

		void onClose();

		void onKeyUp( int _keyCode );
		void onKeyDown( int _keyCode );

		void onMouseMove();
		void onMouseUp();
		void onMouseDown();

/***********************************************************************************/
/** CONSTANTES                                                                    **/
/***********************************************************************************/

	public:

		static const int KC_NONE = 0x0;
		static const int KC_SHIFT = 0x1;	// bit à 1 si la touche shift du clavier est enfoncée
		static const int KC_CTRL = 0x2;		// bit à 1 si la touche ctrl du clavier est enfoncée
		static const int KC_ALT = 0x4;		// bit à 1 si la touche alt du clavier est enfoncée

		static const int MC_NONE = 0x0;
		static const int MC_LEFT = 0x1;		// bit à 1 si le bouton gauche de la souris est enfoncé
		static const int MC_RIGHT = 0x2;	// bit à 1 si le bouton droit de la souris est enfoncé
		static const int MC_MIDDLE = 0x4;	// bit à 1 si le bouton du milieu de la souris est enfoncé

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAWindow( CUDAEngine* _engine )
			Instancie la classe et initialise les ressources internes.
		*/
		CUDAWindow( CUDAEngine* _engine );

		/*
		~CUDAWindow()
			Libère les ressources internes.
		*/
		virtual ~CUDAWindow();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getEngine()
			Renvoie un pointeur vers le moteur associé à cette fenêtre.
		*/
		CUDAEngine* getEngine();

		/*
		setLeft( int _value = 120 )
			Spécifie la position horizontale (en pixels) de la fenêtre à l'écran.
		*/
		void setLeft( int _value = 120 );

		/*
		setLeft( int _value = 80 )
			Spécifie la position verticale (en pixels) de la fenêtre à l'écran.
		*/
		void setTop( int _value = 80 );

		/*
		getLeft()
			Renvoie la position horizontale (en pixels) de la fenêtre à l'écran.
		*/
		int getLeft();

		/*
		getTop()
			Renvoie la position verticale (en pixels) de la fenêtre à l'écran.
		*/
		int getTop();

		/*
		setWidth( int _value = 800 )
			Spécifie la taille horizontale (en pixels) de la fenêtre.
		*/
		void setWidth( int _value = 800 );

		/*
		setHeight( int _value = 600 )
			Spécifie la taille verticale (en pixels) de la fenêtre.
		*/
		void setHeight( int _value = 600 );

		/*
		getWidth()
			Renvoie la taille horizontale (en pixels) de la fenêtre.
		*/
		int getWidth();

		/*
		getHeight()
			Renvoie la taille verticale (en pixels) de la fenêtre.
		*/
		int getHeight();

		/*
		getWidth()
			Renvoie la taille horizontale (en pixels) du cadre interne de la fenêtre.
		*/
		int getInnerWidth();

		/*
		getHeight()
			Renvoie la taille verticale (en pixels) du cadre interne de la fenêtre.
		*/
		int getInnerHeight();

		/*
		isVisible()
			Renvoie true si la fenêtre est visible.
		*/
		bool isVisible();

		/*
		isActive()
			Renvoie true si la fenêtre est active (elle ne l'est plus dès lors que
			l'utilisateur clique sur la croix en haut à droite).
		*/
		bool isActive();

		/*
		registerCloseCallback ( LPCLOSECALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsque la fenêtre est fermée.
		*/
		void registerCloseCallback ( LPCLOSECALLBACK _pCallback );

		/*
		unregisterCloseCallback ( LPCLOSECALLBACK _pCallback )
			Supprime une callback devant être appelée lorsque la fenêtre est fermée.
		*/
		void unregisterCloseCallback ( LPCLOSECALLBACK _pCallback );

		/*
		registerKeyUpCallback ( LPKEYUPCALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsqu'une touche du clavier est relachée.
		*/
		void registerKeyUpCallback ( LPKEYUPCALLBACK _pCallback );

		/*
		unregisterKeyUpCallback ( LPKEYUPCALLBACK _pCallback )
			Supprime une callback devant être appelée lorsqu'une touche du clavier est relachée.
		*/
		void unregisterKeyUpCallback ( LPKEYUPCALLBACK _pCallback );

		/*
		registerKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsqu'une touche du clavier est enfoncée.
		*/
		void registerKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback );

		/*
		unregisterKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback )
			Supprime une callback devant être appelée lorsqu'une touche du clavier est enfoncée.
		*/
		void unregisterKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback );

		/*
		registerMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsque la souris est en mouvement.
		*/
		void registerMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback );

		/*
		unregisterMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback )
			Supprime une callback devant être appelée lorsque la souris est en mouvement.
		*/
		void unregisterMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback );

		/*
		registerMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsqu'un bouton de la souris est relaché.
		*/
		void registerMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback );

		/*
		unregisterMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback )
			Supprime une callback devant être appelée lorsqu'un bouton de la souris est relaché.
		*/
		void unregisterMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback );

		/*
		registerMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback )
			Ajoute une callback devant être appelée lorsqu'un bouton de la souris est enfoncé.
		*/
		void registerMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback );

		/*
		unregisterMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback )
			Supprime une callback devant être appelée lorsqu'un bouton de la souris est enfoncé.
		*/
		void unregisterMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback );

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		show()
			Crée la fenêtre si nécessaire puis l'affiche.
		*/
		virtual void show() = 0;

		/*
		hide()
			Cache la fenêtre.
		*/
		virtual void hide() = 0;

	};
}

/***********************************************************************************/

#endif
