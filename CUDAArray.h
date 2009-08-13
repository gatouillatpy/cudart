
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
			Instancie la classe mais n'alloue pas de m�moire pour son contenu.
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
			Instancie la classe et alloue suffisament de m�moire vid�o pour contenir
			le nombre d'unit�s sp�cifi� par _unit_count. La taille (en octets) de
			chaque unit� est d�duite automatiquement � partir de unit_type.
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
			Instancie la classe et alloue suffisament de m�moire vid�o pour contenir
			le nombre d'unit�s sp�cifi� par _unit_count. La taille (en octets) de
			chaque unit� est sp�cifi�e par _unit_size, ce qui permet par exemple
			de stocker des unit�s de type float3 en conservant un alignement sur 16
			octets, ce qui peut consid�rablement r�duire les temps d'acc�s aux
			donn�es au niveau des noyaux ex�cut�s sur les processeurs de flux.
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
			Lib�re la m�moire vid�o r�serv�e pour ce tableau.
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
			Renvoie le nombre d'unit�s contenues dans ce tableau.
		*/
		int getUnitCount() { return unit_count; }

		/*
		getUnitSize()
			Renvoie la taille d'une unit� de stockage. Attention cette taille
			peut �tre diff�rente de sizeof(unit_type) en cas d'alignement forc�.
		*/
		int getUnitSize() { return unit_size; }

		/*
		getSize()
			Renvoie la taille totale (en nombre d'octets) de m�moire vid�o allou�e
			pour ce tableau.
		*/
		int getSize() { return size; }

		/*
		getPointer()
			Renvoie une pointeur vers la zone m�moire allou�e pour ce tableau.
			Attention cependant, il s'agit d'un pointeur vers de la m�moire vid�o
			et non syst�me. Ce pointeur ne peut donc �tre d�r�f�renc� qu'au sein
			d'un noyau ex�cut� par les processeurs de flux de la carte graphique.
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
			Copie dans ce tableau le contenu de la m�moire syst�me point�e par
			host_ptr. Attention la zone m�moire syst�me point�e par host_ptr doit
			�tre de taille (en octets) �gale ou sup�rieure � celle renvoy�e par
			la m�thode getSize().
		*/
		void copyDataFromHost( unit_type* host_ptr )
		{
			cudaMemcpy( data, host_ptr, size, cudaMemcpyHostToDevice );
		}

		/*
		copyDataToHost( unit_type* host_ptr )
			Copie dans la zone m�moire syst�me point�e par host_ptr le contenu
			de ce tableau. Attention la zone m�moire syst�me point�e par host_ptr
			doit �tre de taille (en octets) �gale ou sup�rieure � celle renvoy�e
			par la m�thode getSize().
		*/
		void copyDataToHost( unit_type* host_ptr )
		{
			cudaMemcpy( host_ptr, data, size, cudaMemcpyDeviceToHost );
		}

		/*
		copyDataFromDevice( unit_type* device_ptr )
			Copie dans ce tableau le contenu de la m�moire vid�o point�e par
			device_ptr. Attention la zone m�moire vid�o point�e par device_ptr
			doit �tre de taille (en octets) �gale ou sup�rieure � celle renvoy�e
			par la m�thode getSize().
		*/
		void copyDataFromDevice( unit_type* device_ptr )
		{
			cudaMemcpy( data, device_ptr, size, cudaMemcpyDeviceToDevice );
		}

		/*
		copyDataToDevice( unit_type* device_ptr )
			Copie dans la zone m�moire vid�o point�e par device_ptr le contenu
			de ce tableau. Attention la zone m�moire vid�o point�e par device_ptr
			doit �tre de taille (en octets) �gale ou sup�rieure � celle renvoy�e
			par la m�thode getSize().
		*/
		void copyDataToDevice( unit_type* device_ptr )
		{
			cudaMemcpy( device_ptr, data, size, cudaMemcpyDeviceToDevice );
		}

	};
}

/***********************************************************************************/

#endif
