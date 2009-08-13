
#ifndef _CUDA_RENDER_SURFACE
#define _CUDA_RENDER_SURFACE

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDASurface.h"

/***********************************************************************************/

namespace renderkit
{
	template <typename unit_type>
	class CUDARenderSurface : public CUDASurface<unit_type>
	{

/***********************************************************************************/
/** CLASSES AMIES                                                                 **/
/***********************************************************************************/

	public:

		friend class CUDAD3DRenderer;
		friend class CUDAOGLRenderer;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDARenderSurface( int _width, int _height )
			Instancie la classe et spécifie la quantité de mémoire vidéo nécessaire
			pour contenir le nombre d'unités spécifié par _width * _height. La taille
			de chaque unité est déduite automatiquement à partir de unit_type.
		*/
		CUDARenderSurface( int _width, int _height )
		{
			_assert( _width > 0, __FILE__, __LINE__, "CUDARenderSurface::CUDARenderSurface() : Invalid parameter '_width', must be positive." );
			_assert( _height > 0, __FILE__, __LINE__, "CUDARenderSurface::CUDARenderSurface() : Invalid parameter '_height', must be positive." );

			this->width = _width;
			this->height = _height;

			this->unit_count = this->width * this->height;
			this->unit_size = sizeof(unit_type);

			this->size = this->unit_count * this->unit_size;

			this->pitch = this->width * this->unit_size;
		}

		/*
		CUDARenderSurface( int _width, int _height, int _unit_size )
			Instancie la classe et spécifie la quantité de mémoire vidéo nécessaire
			pour contenir le nombre d'unités spécifié par _width * _height. La taille
			de chaque unité est spécifiée par _unit_size, ce qui permet par exemple
			de stocker des unités de type float3 en conservant un alignement sur 16
			octets, ce qui peut considérablement réduire les temps d'accès aux
			données au niveau des noyaux exécutés sur les processeurs de flux.
		*/
		CUDARenderSurface( int _width, int _height, int _unit_size )
		{
			_assert( _width > 0, __FILE__, __LINE__, "CUDARenderSurface::CUDARenderSurface() : Invalid parameter '_width', must be positive." );
			_assert( _height > 0, __FILE__, __LINE__, "CUDARenderSurface::CUDARenderSurface() : Invalid parameter '_height', must be positive." );
			_assert( _unit_size > 0, __FILE__, __LINE__, "CUDARenderSurface::CUDARenderSurface() : Invalid parameter '_unit_size', must be positive." );

			this->width = _width;
			this->height = _height;

			this->unit_count = this->width * this->height;
			this->unit_size = _unit_size;

			this->size = this->unit_count * this->unit_size;

			this->pitch = this->width * this->unit_size;
		}

		/*
		~CUDARenderSurface()
			Invalide l'utilisation de la zone mémoire réservée à cette surface.
		*/
		virtual ~CUDARenderSurface()
		{
			this->data = NULL;
		}

	};
}

/***********************************************************************************/

#endif
