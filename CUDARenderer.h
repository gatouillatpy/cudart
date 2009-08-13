
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

		bool fullscreen; // vrai pour le mode plein �cran, faux pour le mode fen�tre

		bool verticalSync; // vrai pour activer la synchronisation verticale

		bool hardware; // vrai pour faire les rendus � l'aide de la carte graphique

		CUDARenderSurface<uint>* surface;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDARenderer( CUDAEngine* _engine )
			Instancie la classe et alloue la m�moire vid�o pour une surface de rendu
			dont la taille correspond � celle de la fen�tre associ�e � _engine.
		*/
		CUDARenderer( CUDAEngine* _engine );

		/*
		~CUDARenderer()
			Lib�re la m�moire vid�o r�serv�e pour une surface de rendu.
		*/
		virtual ~CUDARenderer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		getEngine()
			Renvoie un pointeur vers le moteur associ� � ce renderer.
		*/
		CUDAEngine* getEngine();

		/*
		lockSurface()
			V�rouille l'acc�s � la surface de rendu. L'appel � cette m�thode est
			n�cessaire avant tout acc�s � cette surface.
		*/
		virtual void lockSurface() = 0;

		/*
		unlockSurface()
			D�v�rouille l'acc�s � la surface de rendu. L'appel � cette m�thode est
			n�cessaire avant tout appel � la m�thode update().
		*/
		virtual void unlockSurface() = 0;

		/*
		getSurface()
			Renvoie un pointeur vers la surface de rendu.
		*/
		CUDARenderSurface<uint>* getSurface() { return surface; }

		/*
		enableFullscreen( int _width, int _height )
			Active le mode plein �cran et sp�cifie la r�solution � appliquer.
		*/
		void enableFullscreen( int _width, int _height );

		/*
		disableFullscreen()
			D�sactive le mode plein �cran.
		*/
		void disableFullscreen();

		/*
		enableVSync()
			Active la synchronisation du framerate avec le taux de rafraichissement vertical
			de l'�cran.
		*/
		void enableVSync();

		/*
		disableVSync()
			D�sactive la synchronisation verticale.
		*/
		void disableVSync();

		/*
		useHardware()
			Active l'utilisation de la carte graphique pour le rendu final.
		*/
		void useHardware();

		/*
		useSoftware()
			D�sactive l'utilisation de la carte graphique pour le rendu final.
		*/
		void useSoftware();

		/*
		insertLabel()
			Ajoute un label � afficher au premier plan. Le texte � affich� est d�fini par le
			param�tre _text. Les param�tres _x et _y d�finissent la position en pixels de ce
			label. Enfin les param�tres _r, g_, _b, et _a d�finissent la couleur et la valeur
			de transparence du texte � afficher.
		*/
		void insertLabel( const string& _text, int _x, int _y,
			float _r = 1.0f, float _g = 1.0f, float _b = 1.0f, float _a = 1.0f );

		/*
		clearForeground()
			Supprime tous les �l�ments ajout�s au premier plan.
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
			Lib�re toutes les ressources relatives � l'API.
		*/
		virtual void finalize() = 0;

		/*
		update()
			Met � jour l'affichage en copiant le contenu du back buffer vers le frame buffer.
		*/
		virtual void update() = 0;

		/*
		saveSurface( const char* path, const int format, CUDASurface<byte>* surface )
			Sauvegarde une surface dans un format brut (les donn�es sont �crites telles que
			pr�sentes en m�moire).
		*/
		virtual void saveSurface( const char* path, const int format, CUDASurface<byte>* surface );

		/*
		loadSurface( const char* path )
			Charge un fichier enregistr� dans un format brut.
		*/
		virtual CUDASurface<byte>* loadSurface( const char* path );

	};
}

/***********************************************************************************/

#endif
