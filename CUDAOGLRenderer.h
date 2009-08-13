
#ifndef _CUDA_OGLRENDERER
#define _CUDA_OGLRENDERER

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDARenderer.h"

#include <GL/glew.h>
#include <GL/gl.h>			// Header File For The OpenGL32 Library
#include <GL/glu.h>			// Header File For The GLu32 Library

#ifndef _WIN32
	#include <GL/glx.h>
	#include <X11/extensions/xf86vmode.h>
	#include <X11/keysym.h>
#endif

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDAOGLRenderer : public CUDARenderer
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		#ifdef _WIN32
			HDC hDC;
			HGLRC hRC;
		#else
			GLXContext ctx;
		#endif

		GLuint glRenderTexture;

		void* glHostTextureData;

		GLuint glFontList;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAOGLRenderer( CUDAEngine* _engine )
			Instancie la classe et alloue la mémoire vidéo pour une surface de rendu
			dont la taille correspond à celle de la fenêtre associée à _engine.
		*/
		CUDAOGLRenderer( CUDAEngine* _engine );

		/*
		~CUDAOGLRenderer()
			Libère la mémoire vidéo réservée pour une surface de rendu.
		*/
		virtual ~CUDAOGLRenderer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		lockSurface()
			Vérouille l'accès à la surface de rendu. L'appel à cette méthode est
			nécessaire avant tout accès à cette surface.
		*/
		virtual void lockSurface();

		/*
		unlockSurface()
			Dévérouille l'accès à la surface de rendu. L'appel à cette méthode est
			nécessaire avant tout appel à la méthode update().
		*/
		virtual void unlockSurface();

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		initialize()
			Initialise OpenGL et lui associe la surface de rendu.
		*/
		virtual void initialize();

		/*
		finalize()
			Libère toutes les ressources relatives à OpenGL.
		*/
		virtual void finalize();

		/*
		update()
			Met à jour l'affichage en copiant le contenu du back buffer vers le frame buffer.
		*/
		virtual void update();

	};
}

/***********************************************************************************/

#endif
