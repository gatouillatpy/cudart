
#ifndef _CUDA_SURFACE
#define _CUDA_SURFACE

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDAArray.h"

#include <typeinfo>

/***********************************************************************************/

namespace renderkit
{
	template <typename unit_type>
	class CUDASurface : public CUDAArray<unit_type>
	{

/***********************************************************************************/
/** CLASSES AMIES                                                                 **/
/***********************************************************************************/

	public:

		friend class CUDARenderer;
		friend class CUDAD3DRenderer;
		friend class CUDAOGLRenderer;

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		int width;
		int height;

		int pitch;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDASurface()
			Instancie la classe mais n'alloue pas de mémoire pour son contenu.
		*/
		CUDASurface()
		{
			this->unit_count = 0;
			this->unit_size = 0;

			this->size = 0;

			this->data = NULL;
		}

		/*
		CUDASurface( int _width, int _height )
			Instancie la classe et alloue suffisament de mémoire vidéo pour contenir
			le nombre d'unités spécifié par _width * _height. La taille (en octets)
			de chaque unité est déduite automatiquement à partir de unit_type.
		*/
		CUDASurface( int _width, int _height )
		{
			_assert( _width > 0, __FILE__, __LINE__, "CUDASurface::CUDASurface() : Invalid parameter '_width', must be positive." );
			_assert( _height > 0, __FILE__, __LINE__, "CUDASurface::CUDASurface() : Invalid parameter '_height', must be positive." );

			width = _width;
			height = _height;

			this->unit_count = width * height;
			this->unit_size = sizeof(unit_type);

			this->size = this->unit_count * this->unit_size;

			pitch = width * this->unit_size;

			this->initialize();
		}

		/*
		CUDASurface( int _width, int _height, int _unit_size )
			Instancie la classe et alloue suffisament de mémoire vidéo pour contenir
			le nombre d'unités spécifié par _width * _height. La taille (en octets)
			de chaque unité est spécifiée par _unit_size, ce qui permet par exemple
			de stocker des unités de type float3 en conservant un alignement sur 16
			octets, ce qui peut considérablement réduire les temps d'accès aux
			données au niveau des noyaux exécutés sur les processeurs de flux.
		*/
		CUDASurface( int _width, int _height, int _unit_size )
		{
			_assert( _width > 0, __FILE__, __LINE__, "CUDASurface::CUDASurface() : Invalid parameter '_width', must be positive." );
			_assert( _height > 0, __FILE__, __LINE__, "CUDASurface::CUDASurface() : Invalid parameter '_height', must be positive." );
			_assert( _unit_size > 0, __FILE__, __LINE__, "CUDASurface::CUDASurface() : Invalid parameter '_unit_size', must be positive." );

			width = _width;
			height = _height;

			this->unit_count = width * height;
			this->unit_size = _unit_size;

			this->size = this->unit_count * this->unit_size;

			pitch = width * this->unit_size;

			this->initialize();
		}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getWidth()
			Renvoie le nombre de points par ligne contenues dans cette surface.
		*/
		int getWidth() { return width; }

		/*
		getHeight()
			Renvoie le nombre de lignes au total contenues dans cette surface.
		*/
		int getHeight() { return height; }

		/*
		getPitch()
			Renvoie la taille d'une ligne (en octets) de cette surface.
		*/
		int getPitch() { return pitch; }

	};
}

/***********************************************************************************/

#endif
