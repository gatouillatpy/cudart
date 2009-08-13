
#ifdef _WIN32

#ifndef _CUDA_D3DRENDERER
#define _CUDA_D3DRENDERER

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <windows.h>

#include <d3dx9.h>
#include <dxerr.h>

#include "CUDARenderer.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDAD3DRenderer : public CUDARenderer
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		LPDIRECT3D9 lpD3D;
		LPDIRECT3DDEVICE9 lpD3DDevice;

		LPDIRECT3DTEXTURE9 lpRenderTexture;

		LPDIRECT3DPIXELSHADER9 lpPixelShader;
		LPDIRECT3DVERTEXSHADER9 lpVertexShader;

		LPDIRECT3DVERTEXDECLARATION9 lpVertexDeclaration;

		LPDIRECT3DVERTEXBUFFER9 lpVertexBuffer;
		LPDIRECT3DINDEXBUFFER9 lpIndexBuffer;

		LPD3DXFONT lpFont;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAD3DRenderer( CUDAEngine* _engine )
			Instancie la classe et alloue la m�moire vid�o pour une surface de rendu
			dont la taille correspond � celle de la fen�tre associ�e � _engine.
		*/
		CUDAD3DRenderer( CUDAEngine* _engine );

		/*
		~CUDAD3DRenderer()
			Lib�re la m�moire vid�o r�serv�e pour une surface de rendu.
		*/
		virtual ~CUDAD3DRenderer();

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
			Initialise Direct3D et lui associe la surface de rendu.
		*/
		virtual void initialize();

		/*
		finalize()
			Lib�re toutes les ressources relatives � Direct3D.
		*/
		virtual void finalize();

		/*
		update()
			Met � jour l'affichage en copiant le contenu du back buffer vers le frame buffer.
		*/
		virtual void update();

		/*
		saveSurface( const char* path, const int format, CUDASurface<byte>* surface )
			Sauvegarde une surface dans l'un des formats reconnus par D3DX.
		*/
		virtual void saveSurface( const char* path, const int format, CUDASurface<byte>* surface );

		/*
		loadSurface( const char* path )
			Charge un fichier dont le format est reconnu par D3DX.
		*/
		virtual CUDASurface<byte>* loadSurface( const char* path );

	};
}

/***********************************************************************************/

#endif

#endif
