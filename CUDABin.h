
#ifndef _CUDA_BIN
#define _CUDA_BIN

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

/***********************************************************************************/

namespace renderkit
{
	struct __builtin_align__(16) bvbin
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

		float3 box_min;
		float3 box_max;

		int obj_count;

		int left_count;
		int right_count;

		float left_area;
		float right_area;

		int padding; // pour conserver l'alignement sur 16 octets en mode deviceemu
	    
/***********************************************************************************/
/** METHODES                                                                      **/
/***********************************************************************************/

		/*
		reset()
			Réinitialise ce compartiment avec notamment un nombre d'objets contenu
			nul ainsi qu'une boite englobante "négative" (min > max).
		*/
		inline __device__ __host__ void reset()
		{
			box_min = make_float3( +CUDART_NORM_HUGE_F );
			box_max = make_float3( -CUDART_NORM_HUGE_F );

			obj_count = 0;

			left_count = 0; 
			right_count = 0;

			left_area = 0.f;
			right_area = 0.f;
		}
	    
		/*
		insertObject()
			Ajoute un objet à ce compartiment et met à jour sa boite englobante
			en fonction de celle de l'objet en question.
		*/
		inline __device__ __host__ void insertObject( const aabox& _box )
		{
			box_min = fminf( box_min, _box.min );
			box_max = fmaxf( box_max, _box.max );

			obj_count++;
		}

		/*
		insertLeft( const int _count, const aabox& _box )
			Ajoute un ou plusieurs objets à gauche de ce compartiment et met à jour
			la surface de l'englobant des objets situés à gauche en fonction de la
			boite de passée en paramètre.
		*/
		inline __device__ __host__ void insertLeft( const int _count, const aabox& _box )
		{
			left_area += _box.getArea();

			left_count += _count;
		}
	    
		/*
		insertRight( const int _count, const aabox& _box )
			Ajoute un ou plusieurs objets à droite de ce compartiment et met à jour
			la surface de l'englobant des objets situés à droite en fonction de la
			boite de passée en paramètre.
		*/
		inline __device__ __host__ void insertRight( const int _count, const aabox& _box )
		{
			right_area += _box.getArea();

			right_count += _count;
		}
	    
		/*
		getSAH( const aabox& parent, const float box_cost = 1.f, const float obj_cost = 1.f )
			Renvoie le facteur heuristique de ce compartiement en fonction de la
			surface de l'englobant des objets situés de part et d'autre ainsi que
			de la surface de l'englobant du noeud parent et des couts d'intersection
			d'une boite ou d'un objet.
		*/
		inline __device__ __host__ float getSAH( const aabox& parent, const float box_cost = 1.f, const float obj_cost = 1.f ) const
		{
			const float parent_area = parent.getArea();

			return box_cost * 2.f + obj_cost
						* ( (left_area / parent_area) * left_count
						  + (right_area / parent_area) * right_count );
		}

	}; // 48 octets

	typedef bvbin CUDABin;

}

/***********************************************************************************/

#endif
