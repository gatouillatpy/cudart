
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
			Instancie la classe et alloue la m�moire vid�o pour une surface de rendu
			dont la taille correspond � celle de la fen�tre associ�e � _engine.
		*/
		CUDAOGLRenderer( CUDAEngine* _engine );

		/*
		~CUDAOGLRenderer()
			Lib�re la m�moire vid�o r�serv�e pour une surface de rendu.
		*/
		virtual ~CUDAOGLRenderer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		lockSurface()
			V�rouille l'acc�s � la surface de rendu. L'appel � cette m�thode est
			n�cessaire avant tout acc�s � cette surface.
		*/
		virtual void lockSurface();

		/*
		unlockSurface()
			D�v�rouille l'acc�s � la surface de rendu. L'appel � cette m�thode est
			n�cessaire avant tout appel � la m�thode update().
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
			Lib�re toutes les ressources relatives � OpenGL.
		*/
		virtual void finalize();

		/*
		update()
			Met � jour l'affichage en copiant le contenu du back buffer vers le frame buffer.
		*/
		virtual void update();

	};
}

/***********************************************************************************/

#endif
