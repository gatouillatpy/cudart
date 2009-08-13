
#ifndef _CUDA_RENDERER
#define _CUDA_RENDERER

#pragma warning (disable : 4996)

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDARenderSurface.h"

#include <vector>
#include <string>

using namespace std;

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	const int CUDA_FORMAT_BMP = 0;
	const int CUDA_FORMAT_JPG = 1;
	const int CUDA_FORMAT_TGA = 2;
	const int CUDA_FORMAT_PNG = 3;
	const int CUDA_FORMAT_DDS = 4;
	const int CUDA_FORMAT_PPM = 5;
	const int CUDA_FORMAT_DIB = 6;
	const int CUDA_FORMAT_HDR = 7;
	const int CUDA_FORMAT_PFM = 8;
	const int CUDA_FORMAT_RAW = 9;

	class CUDARenderer
	{

/***********************************************************************************/
/** TYPES                                                                         **/
/***********************************************************************************/

	public:

		typedef struct
		{
			string text;
			int x, y;
			float r, g, b, a;
		} CUDALabel;

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		CUDAEngine* engine;

		vector<CUDALabel> labels;

		bool fullscreen; // vrai pour le mode plein écran, faux pour le mode fenêtre

		bool verticalSync; // vrai pour activer la synchronisation verticale

		bool hardware; // vrai pour faire les rendus à l'aide de la carte graphique

		CUDARenderSurface<uint>* surface;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDARenderer( CUDAEngine* _engine )
			Instancie la classe et alloue la mémoire vidéo pour une surface de rendu
			dont la taille correspond à celle de la fenêtre associée à _engine.
		*/
		CUDARenderer( CUDAEngine* _engine );

		/*
		~CUDARenderer()
			Libère la mémoire vidéo réservée pour une surface de rendu.
		*/
		virtual ~CUDARenderer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		getEngine()
			Renvoie un pointeur vers le moteur associé à ce renderer.
		*/
		CUDAEngine* getEngine();

		/*
		lockSurface()
			Vérouille l'accès à la surface de rendu. L'appel à cette méthode est
			nécessaire avant tout accès à cette surface.
		*/
		virtual void lockSurface() = 0;

		/*
		unlockSurface()
			Dévérouille l'accès à la surface de rendu. L'appel à cette méthode est
			nécessaire avant tout appel à la méthode update().
		*/
		virtual void unlockSurface() = 0;

		/*
		getSurface()
			Renvoie un pointeur vers la surface de rendu.
		*/
		CUDARenderSurface<uint>* getSurface() { return surface; }

		/*
		enableFullscreen( int _width, int _height )
			Active le mode plein écran et spécifie la résolution à appliquer.
		*/
		void enableFullscreen( int _width, int _height );

		/*
		disableFullscreen()
			Désactive le mode plein écran.
		*/
		void disableFullscreen();

		/*
		enableVSync()
			Active la synchronisation du framerate avec le taux de rafraichissement vertical
			de l'écran.
		*/
		void enableVSync();

		/*
		disableVSync()
			Désactive la synchronisation verticale.
		*/
		void disableVSync();

		/*
		useHardware()
			Active l'utilisation de la carte graphique pour le rendu final.
		*/
		void useHardware();

		/*
		useSoftware()
			Désactive l'utilisation de la carte graphique pour le rendu final.
		*/
		void useSoftware();

		/*
		insertLabel()
			Ajoute un label à afficher au premier plan. Le texte à affiché est défini par le
			paramètre _text. Les paramètres _x et _y définissent la position en pixels de ce
			label. Enfin les paramètres _r, g_, _b, et _a définissent la couleur et la valeur
			de transparence du texte à afficher.
		*/
		void insertLabel( const string& _text, int _x, int _y,
			float _r = 1.0f, float _g = 1.0f, float _b = 1.0f, float _a = 1.0f );

		/*
		clearForeground()
			Supprime tous les éléments ajoutés au premier plan.
		*/
		void clearForeground();

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		initialize()
			Initialise l'API et lui associe la surface de rendu.
		*/
		virtual void initialize() = 0;

		/*
		finalize()
			Libère toutes les ressources relatives à l'API.
		*/
		virtual void finalize() = 0;

		/*
		update()
			Met à jour l'affichage en copiant le contenu du back buffer vers le frame buffer.
		*/
		virtual void update() = 0;

		/*
		saveSurface( const char* path, const int format, CUDASurface<byte>* surface )
			Sauvegarde une surface dans un format brut (les données sont écrites telles que
			présentes en mémoire).
		*/
		virtual void saveSurface( const char* path, const int format, CUDASurface<byte>* surface );

		/*
		loadSurface( const char* path )
			Charge un fichier enregistré dans un format brut.
		*/
		virtual CUDASurface<byte>* loadSurface( const char* path );

	};
}

/***********************************************************************************/

#endif
