
#ifndef _CUDA_ARRAY
#define _CUDA_ARRAY

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <cuda_runtime.h>

/***********************************************************************************/

namespace renderkit
{
	template <typename unit_type>
	class CUDAArray
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		int unit_count;
		int unit_size;

		int size;

		unit_type* data;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAArray()
			Instancie la classe mais n'alloue pas de mémoire pour son contenu.
		*/
		CUDAArray()
		{
			unit_count = 0;
			unit_size = sizeof(unit_type);

			size = 0;

			data = NULL;
		}

		/*
		CUDAArray( int _unit_count )
			Instancie la classe et alloue suffisament de mémoire vidéo pour contenir
			le nombre d'unités spécifié par _unit_count. La taille (en octets) de
			chaque unité est déduite automatiquement à partir de unit_type.
		*/
		CUDAArray( int _unit_count )
		{
			_assert( _unit_count > 0, __FILE__, __LINE__, "CUDAArray::CUDAArray() : Invalid parameter '_unit_count', must be positive." );

			unit_count = _unit_count;
			unit_size = sizeof(unit_type);

			size = unit_count * unit_size;

			initialize();
		}

		/*
		CUDAArray( int _unit_count, int _unit_size )
			Instancie la classe et alloue suffisament de mémoire vidéo pour contenir
			le nombre d'unités spécifié par _unit_count. La taille (en octets) de
			chaque unité est spécifiée par _unit_size, ce qui permet par exemple
			de stocker des unités de type float3 en conservant un alignement sur 16
			octets, ce qui peut considérablement réduire les temps d'accès aux
			données au niveau des noyaux exécutés sur les processeurs de flux.
		*/
		CUDAArray( int _unit_count, int _unit_size )
		{
			_assert( _unit_count > 0, __FILE__, __LINE__, "CUDAArray::CUDAArray() : Invalid parameter '_unit_count', must be positive." );
			_assert( _unit_size > 0, __FILE__, __LINE__, "CUDAArray::CUDAArray() : Invalid parameter '_unit_size', must be positive." );

			unit_count = _unit_count;
			unit_size = _unit_size;

			size = unit_count * unit_size;

			initialize();
		}

		/*
		~CUDAArray()
			Libère la mémoire vidéo réservée pour ce tableau.
		*/
		virtual ~CUDAArray()
		{
			finalize();
		}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getUnitCount()
			Renvoie le nombre d'unités contenues dans ce tableau.
		*/
		int getUnitCount() { return unit_count; }

		/*
		getUnitSize()
			Renvoie la taille d'une unité de stockage. Attention cette taille
			peut être différente de sizeof(unit_type) en cas d'alignement forcé.
		*/
		int getUnitSize() { return unit_size; }

		/*
		getSize()
			Renvoie la taille totale (en nombre d'octets) de mémoire vidéo allouée
			pour ce tableau.
		*/
		int getSize() { return size; }

		/*
		getPointer()
			Renvoie une pointeur vers la zone mémoire allouée pour ce tableau.
			Attention cependant, il s'agit d'un pointeur vers de la mémoire vidéo
			et non système. Ce pointeur ne peut donc être déréférencé qu'au sein
			d'un noyau exécuté par les processeurs de flux de la carte graphique.
		*/
		unit_type* getPointer() { return data; }

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	protected:

		void initialize()
		{
			_assert( cudaMalloc( (void**)&data, size ) == cudaSuccess, __FILE__, __LINE__,
						"CUDAArray::initialize() : Unable to allocate enough video memory." );
		}

		void finalize()
		{
			if ( data )
				cudaFree( data );
		}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		copyDataFromHost( unit_type* host_ptr )
			Copie dans ce tableau le contenu de la mémoire système pointée par
			host_ptr. Attention la zone mémoire système pointée par host_ptr doit
			être de taille (en octets) égale ou supérieure à celle renvoyée par
			la méthode getSize().
		*/
		void copyDataFromHost( unit_type* host_ptr )
		{
			cudaMemcpy( data, host_ptr, size, cudaMemcpyHostToDevice );
		}

		/*
		copyDataToHost( unit_type* host_ptr )
			Copie dans la zone mémoire système pointée par host_ptr le contenu
			de ce tableau. Attention la zone mémoire système pointée par host_ptr
			doit être de taille (en octets) égale ou supérieure à celle renvoyée
			par la méthode getSize().
		*/
		void copyDataToHost( unit_type* host_ptr )
		{
			cudaMemcpy( host_ptr, data, size, cudaMemcpyDeviceToHost );
		}

		/*
		copyDataFromDevice( unit_type* device_ptr )
			Copie dans ce tableau le contenu de la mémoire vidéo pointée par
			device_ptr. Attention la zone mémoire vidéo pointée par device_ptr
			doit être de taille (en octets) égale ou supérieure à celle renvoyée
			par la méthode getSize().
		*/
		void copyDataFromDevice( unit_type* device_ptr )
		{
			cudaMemcpy( data, device_ptr, size, cudaMemcpyDeviceToDevice );
		}

		/*
		copyDataToDevice( unit_type* device_ptr )
			Copie dans la zone mémoire vidéo pointée par device_ptr le contenu
			de ce tableau. Attention la zone mémoire vidéo pointée par device_ptr
			doit être de taille (en octets) égale ou supérieure à celle renvoyée
			par la méthode getSize().
		*/
		void copyDataToDevice( unit_type* device_ptr )
		{
			cudaMemcpy( device_ptr, data, size, cudaMemcpyDeviceToDevice );
		}

	};
}

/***********************************************************************************/

#endif
