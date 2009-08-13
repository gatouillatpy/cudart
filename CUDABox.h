
#ifndef _CUDA_HITBOX
#define _CUDA_HITBOX

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <math_constants.h>

/***********************************************************************************/

namespace renderkit
{
	struct __builtin_align__(16) aabox
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

		float4 min;
		float4 max;

/***********************************************************************************/
/** METHODES                                                                      **/
/***********************************************************************************/

		/*
		getArea()
			Renvoie la surface (en u²) calculée pour cette boite englobante.
		*/
		inline __device__ __host__ float getArea() const
		{
			float4 v = max - min;

			const float area = 2.f * ( v.x * v.y
									 + v.x * v.z
									 + v.y * v.z );

			return area;
		}

		/*
		reset()
			Réinitialise cette boite englobante comme "négative" (min > max).
		*/
		inline __device__ __host__ void reset()
		{
			min = make_float4( +CUDART_NORM_HUGE_F );
			max = make_float4( -CUDART_NORM_HUGE_F );
		}
	    
		/*
		reset( const float4& v )
			Réinitialise cette boite englobante comme "ponctuelle" en fonction du
			point passé en paramètre.
		*/
		inline __device__ __host__ void reset( const float4& v )
		{
			min = v;
			max = v;
		}

		/*
		reset( const float4& _min, const float4& _max )
			Réinitialise cette boite englobante à l'aide des points minimum et
			maximum passés en paramètre.
		*/
		inline __device__ __host__ void reset( const float4& _min, const float4& _max )
		{
			min = _min;
			max = _max;
		}

		/*
		reset( const float3& _min, const float3& _max )
			Réinitialise cette boite englobante à l'aide des points minimum et
			maximum passés en paramètre.
		*/
		inline __device__ __host__ void reset( const float3& _min, const float3& _max )
		{
			min = make_float4( _min );
			max = make_float4( _max );
		}

		/*
		merge( const aabox& b )
			Met à jour cette boite en englobant celle passée en paramètre.
		*/
		inline __device__ __host__ void merge( const aabox& b )
		{
			min = fminf( min, b.min );
			max = fmaxf( max, b.max );
		}

		/*
		merge( const float4& v )
			Met à jour cette boite en englobant le point passé en paramètre.
		*/
		inline __device__ __host__ void merge( const float4& v )
		{
			min = fminf( min, v );
			max = fmaxf( max, v );
		}

		/*
		merge( const float4& _min, const float4& _max )
			Met à jour cette boite en englobant les points minimum et maximum
			passés en paramètre.
		*/
		inline __device__ __host__ void merge( const float4& _min, const float4& _max )
		{
			min = fminf( min, _min );
			max = fmaxf( max, _max );
		}

		/*
		merge( const float3& _min, const float3& _max )
			Met à jour cette boite en englobant les points minimum et maximum
			passés en paramètre.
		*/
		inline __device__ __host__ void merge( const float3& _min, const float3& _max )
		{
			min = fminf( min, _min );
			max = fmaxf( max, _max );
		}

/***********************************************************************************/
/** OPERATEURS                                                                    **/
/***********************************************************************************/

		/*
		operator=( const aabox& b )
			Copie dans cette boite englobante le contenu de celle située à droite
			de l'opérateur =.
		*/
		inline __device__ __host__ aabox& operator=( const aabox& b )
		{
			min = b.min;
			max = b.max;

			return *this;
		}
	    
	}; // 32 octets

	typedef aabox CUDABox;

}

/***********************************************************************************/

#endif
