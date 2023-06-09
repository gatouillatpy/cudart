
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
		static const int KC_SHIFT = 0x1;	// bit � 1 si la touche shift du clavier est enfonc�e
		static const int KC_CTRL = 0x2;		// bit � 1 si la touche ctrl du clavier est enfonc�e
		static const int KC_ALT = 0x4;		// bit � 1 si la touche alt du clavier est enfonc�e

		static const int MC_NONE = 0x0;
		static const int MC_LEFT = 0x1;		// bit � 1 si le bouton gauche de la souris est enfonc�
		static const int MC_RIGHT = 0x2;	// bit � 1 si le bouton droit de la souris est enfonc�
		static const int MC_MIDDLE = 0x4;	// bit � 1 si le bouton du milieu de la souris est enfonc�

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
			Lib�re les ressources internes.
		*/
		virtual ~CUDAWindow();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getEngine()
			Renvoie un pointeur vers le moteur associ� � cette fen�tre.
		*/
		CUDAEngine* getEngine();

		/*
		setLeft( int _value = 120 )
			Sp�cifie la position horizontale (en pixels) de la fen�tre � l'�cran.
		*/
		void setLeft( int _value = 120 );

		/*
		setLeft( int _value = 80 )
			Sp�cifie la position verticale (en pixels) de la fen�tre � l'�cran.
		*/
		void setTop( int _value = 80 );

		/*
		getLeft()
			Renvoie la position horizontale (en pixels) de la fen�tre � l'�cran.
		*/
		int getLeft();

		/*
		getTop()
			Renvoie la position verticale (en pixels) de la fen�tre � l'�cran.
		*/
		int getTop();

		/*
		setWidth( int _value = 800 )
			Sp�cifie la taille horizontale (en pixels) de la fen�tre.
		*/
		void setWidth( int _value = 800 );

		/*
		setHeight( int _value = 600 )
			Sp�cifie la taille verticale (en pixels) de la fen�tre.
		*/
		void setHeight( int _value = 600 );

		/*
		getWidth()
			Renvoie la taille horizontale (en pixels) de la fen�tre.
		*/
		int getWidth();

		/*
		getHeight()
			Renvoie la taille verticale (en pixels) de la fen�tre.
		*/
		int getHeight();

		/*
		getWidth()
			Renvoie la taille horizontale (en pixels) du cadre interne de la fen�tre.
		*/
		int getInnerWidth();

		/*
		getHeight()
			Renvoie la taille verticale (en pixels) du cadre interne de la fen�tre.
		*/
		int getInnerHeight();

		/*
		isVisible()
			Renvoie true si la fen�tre est visible.
		*/
		bool isVisible();

		/*
		isActive()
			Renvoie true si la fen�tre est active (elle ne l'est plus d�s lors que
			l'utilisateur clique sur la croix en haut � droite).
		*/
		bool isActive();

		/*
		registerCloseCallback ( LPCLOSECALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsque la fen�tre est ferm�e.
		*/
		void registerCloseCallback ( LPCLOSECALLBACK _pCallback );

		/*
		unregisterCloseCallback ( LPCLOSECALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsque la fen�tre est ferm�e.
		*/
		void unregisterCloseCallback ( LPCLOSECALLBACK _pCallback );

		/*
		registerKeyUpCallback ( LPKEYUPCALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsqu'une touche du clavier est relach�e.
		*/
		void registerKeyUpCallback ( LPKEYUPCALLBACK _pCallback );

		/*
		unregisterKeyUpCallback ( LPKEYUPCALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsqu'une touche du clavier est relach�e.
		*/
		void unregisterKeyUpCallback ( LPKEYUPCALLBACK _pCallback );

		/*
		registerKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsqu'une touche du clavier est enfonc�e.
		*/
		void registerKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback );

		/*
		unregisterKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsqu'une touche du clavier est enfonc�e.
		*/
		void unregisterKeyDownCallback ( LPKEYDOWNCALLBACK _pCallback );

		/*
		registerMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsque la souris est en mouvement.
		*/
		void registerMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback );

		/*
		unregisterMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsque la souris est en mouvement.
		*/
		void unregisterMouseMoveCallback ( LPMOUSEMOVECALLBACK _pCallback );

		/*
		registerMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsqu'un bouton de la souris est relach�.
		*/
		void registerMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback );

		/*
		unregisterMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsqu'un bouton de la souris est relach�.
		*/
		void unregisterMouseUpCallback ( LPMOUSEUPCALLBACK _pCallback );

		/*
		registerMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback )
			Ajoute une callback devant �tre appel�e lorsqu'un bouton de la souris est enfonc�.
		*/
		void registerMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback );

		/*
		unregisterMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback )
			Supprime une callback devant �tre appel�e lorsqu'un bouton de la souris est enfonc�.
		*/
		void unregisterMouseDownCallback ( LPMOUSEDOWNCALLBACK _pCallback );

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		show()
			Cr�e la fen�tre si n�cessaire puis l'affiche.
		*/
		virtual void show() = 0;

		/*
		hide()
			Cache la fen�tre.
		*/
		virtual void hide() = 0;

	};
}

/***********************************************************************************/

#endif
