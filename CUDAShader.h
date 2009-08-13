
#ifndef _CUDA_SHADER
#define _CUDA_SHADER

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDARenderSurface.h"
#include "CUDABuffer.h"
#include "CUDACamera.h"

using namespace std;

/***********************************************************************************/

namespace renderkit
{
	class CUDAShader
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		float4 backColor; // couleur de l'arrière plan

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAShader()
			Instancie cette classe et définit les valeurs par défaut pour les
			paramètres de ce shader.
		*/
		CUDAShader();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getBackColor()
			Renvoie la couleur d'arrière plan du rendu final.
		*/
		float4 getBackColor() { return backColor; }

		/*
		setBackColor()
			Spécifie la couleur d'arrière plan du rendu final.
		*/
		void setBackColor( float _r = 0.0f, float _g = 0.0f,
			float _b = 0.0f, float _a = 0.0f )
		{
			backColor.x = _r;
			backColor.y = _g;
			backColor.z = _b;
			backColor.w = _a;
		}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		run( CUDARenderSurface<uint>* output, CUDABuffer* input, CUDACamera* camera )
			Applique ce shader et renvoie le rendu final dans la surface pointée par
			le paramètre output.
		*/
		virtual void run( CUDARenderSurface<uint>* output, CUDABuffer* input, CUDACamera* camera );

	};
}

/***********************************************************************************/

#endif
