
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

#include <vector_types.h>
#include <vector_functions.h>

#include <math.h>
#include <time.h>

#include <string>
#include <vector>
#include <algorithm>

#pragma warning (disable : 4996)

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "../CUDADebug.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void CUDA_radixSort( uint* values, uint length, uint bit_depth = 32 );

void CUDA_radixSortGeneral( uint* keys, uint* indices, uint length, uint bit_depth = 32, bool init_indices = false );

template <typename value_type>
void CUDA_bitonicSort( value_type* values, uint length );

template <typename key_type, typename value_type>
void CUDA_bitonicSortGeneral( value_type* keys, value_type* values, uint length );

/***********************************************************************************/

using namespace std;

/***********************************************************************************/
/** VARIABLES GLOBALES                                                            **/
/***********************************************************************************/

const uint test_size = 1397121;

uint* test_sort_indices;
uint* test_sort_indices_host;
uint* test_sort_keys;
uint* test_sort_keys_host;
uint* test_sort_values;
uint* test_sort_values_host;

cudaEvent_t e_start, e_stop;

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void testSTLSort()
{
	debugPrint( "Test du tri STL pour %d valeurs :\n", test_size );

	vector<uint> test_sort_vector;

	test_sort_vector.reserve( test_size * sizeof(uint) );

	for ( uint k = 0 ; k < test_size ; k++ )
	{
		test_sort_vector.push_back( ( rand() * RAND_MAX + rand() ) % 2097152 );
	}

	cudaEventRecord( e_start, 0 );
	cudaEventSynchronize( e_start );

	sort( test_sort_vector.begin(), test_sort_vector.end() );

	cudaEventRecord( e_stop, 0 );
	cudaEventSynchronize( e_stop );

	float time; cudaEventElapsedTime( &time, e_start, e_stop );

	debugPrint( ">> tri en %.3f ms\n", time );

	for ( uint k = 1 ; k < test_size ; k++ )
	{
		assert( test_sort_vector[k] >= test_sort_vector[k-1] );

		if ( k % 100000 == 0 )
			debugPrint( "%d\n", test_sort_vector[k] );
	}
}

void testBitonicSort()
{
	debugPrint( "Test du tri bitonic pour %d valeurs :\n", test_size );

	for ( uint k = 0 ; k < test_size ; k++ )
	{
		test_sort_values_host[k] = ( rand() * RAND_MAX + rand() ) % 2097152;
	}

	cudaMemcpy( test_sort_values, test_sort_values_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );

	cudaEventRecord( e_start, 0 );

	CUDA_bitonicSort<uint>( test_sort_values, test_size );

	cudaEventRecord( e_stop, 0 );
	cudaEventSynchronize( e_stop );

	float time; cudaEventElapsedTime( &time, e_start, e_stop );

	debugPrint( ">> tri en %.3f ms\n", time );

	cudaMemcpy( test_sort_values_host, test_sort_values, test_size * sizeof(uint), cudaMemcpyDeviceToHost );

	for ( uint k = 1 ; k < test_size ; k++ )
	{
		assert( test_sort_values_host[k] >= test_sort_values_host[k-1] );

		if ( k % 100000 == 0 )
			debugPrint( "%d\n", test_sort_values_host[k] );
	}
}

void testRadixSort()
{
	debugPrint( "Test du tri radix pour %d valeurs :\n", test_size );

	for ( uint k = 0 ; k < test_size ; k++ )
	{
		test_sort_values_host[k] = ( rand() * RAND_MAX + rand() ) % 2097152;
	}

	cudaMemcpy( test_sort_values, test_sort_values_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );

	cudaEventRecord( e_start, 0 );

	CUDA_radixSort( test_sort_values, test_size, 21 );

	cudaEventRecord( e_stop, 0 );
	cudaEventSynchronize( e_stop );

	float time; cudaEventElapsedTime( &time, e_start, e_stop );

	debugPrint( ">> tri en %.3f ms\n", time );

	cudaMemcpy( test_sort_values_host, test_sort_values, test_size * sizeof(uint), cudaMemcpyDeviceToHost );

	for ( uint k = 1 ; k < test_size ; k++ )
	{
		assert( test_sort_values_host[k] >= test_sort_values_host[k-1] );

		if ( k % 100000 == 0 )
			debugPrint( "%d\n", test_sort_values_host[k] );
	}
}

void testBitonicSortGeneral()
{
	debugPrint( "Test du tri bitonic general pour %d valeurs :\n", test_size );

	for ( uint k = 0 ; k < test_size ; k++ )
	{
		test_sort_values_host[k] = ( rand() * RAND_MAX + rand() ) % 2097152;
		test_sort_keys_host[k] = test_sort_values_host[k];
	}

	cudaMemcpy( test_sort_values, test_sort_values_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );
	cudaMemcpy( test_sort_keys, test_sort_keys_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );

	cudaEventRecord( e_start, 0 );

	CUDA_bitonicSortGeneral<uint,uint>( test_sort_keys, test_sort_values, test_size );

	cudaEventRecord( e_stop, 0 );
	cudaEventSynchronize( e_stop );

	float time; cudaEventElapsedTime( &time, e_start, e_stop );

	debugPrint( ">> tri en %.3f ms\n", time );

	cudaMemcpy( test_sort_values_host, test_sort_values, test_size * sizeof(uint), cudaMemcpyDeviceToHost );
	cudaMemcpy( test_sort_keys_host, test_sort_keys, test_size * sizeof(uint), cudaMemcpyDeviceToHost );

	for ( uint k = 1 ; k < test_size ; k++ )
	{
		assert( test_sort_values_host[k]
		>= test_sort_values_host[k-1] );

		if ( k % 100000 == 0 )
			debugPrint( "%d\n", test_sort_values_host[k] );
	}
}

void testRadixSortGeneral()
{
	debugPrint( "Test du tri radix general pour %d valeurs :\n", test_size );

	for ( uint k = 0 ; k < test_size ; k++ )
	{
		test_sort_values_host[k] = ( rand() * RAND_MAX + rand() ) % 2097152;
		test_sort_keys_host[k] = test_sort_values_host[k];
		test_sort_indices_host[k] = k;
	}

	cudaMemcpy( test_sort_keys, test_sort_keys_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );
	cudaMemcpy( test_sort_indices, test_sort_indices_host, test_size * sizeof(uint), cudaMemcpyHostToDevice );

	cudaEventRecord( e_start, 0 );

	CUDA_radixSortGeneral( test_sort_keys, test_sort_indices, test_size, 21, false );

	cudaEventRecord( e_stop, 0 );
	cudaEventSynchronize( e_stop );

	float time; cudaEventElapsedTime( &time, e_start, e_stop );

	debugPrint( ">> tri en %.3f ms\n", time );

	cudaMemcpy( test_sort_keys_host, test_sort_keys, test_size * sizeof(uint), cudaMemcpyDeviceToHost );
	cudaMemcpy( test_sort_indices_host, test_sort_indices, test_size * sizeof(uint), cudaMemcpyDeviceToHost );

	for ( uint k = 1 ; k < test_size ; k++ )
	{
		assert( test_sort_values_host[test_sort_indices_host[k]]
		>= test_sort_values_host[test_sort_indices_host[k-1]] );

		if ( k % 100000 == 0 )
			debugPrint( "%d\n", test_sort_values_host[test_sort_indices_host[k]] );
	}
}

/***********************************************************************************/
/** POINT D'ENTREE                                                                **/
/***********************************************************************************/

int main( int argc, char **argv )
{
	// initialisation des tris

	srand( (uint)time( NULL ) );

	cudaMalloc( (void**)&test_sort_keys, test_size * sizeof(uint) );
	cudaMalloc( (void**)&test_sort_values, test_size * sizeof(uint) );
	cudaMalloc( (void**)&test_sort_indices, test_size * sizeof(uint) );
	cudaMallocHost( (void**)&test_sort_keys_host, test_size * sizeof(uint) );
	cudaMallocHost( (void**)&test_sort_values_host, test_size * sizeof(uint) );
	cudaMallocHost( (void**)&test_sort_indices_host, test_size * sizeof(uint) );

	cudaEventCreate( &e_start );
	cudaEventCreate( &e_stop );

	// tri de valeurs dans un tableau

	testSTLSort();

	testBitonicSort();

	testRadixSort();

	// tri de valeurs dans un tableau en fonction de clés

	testBitonicSortGeneral();

	// tri d'indices dans un tableau en fonction de clés

	testRadixSortGeneral();

	// libération de la mémoire

	cudaEventDestroy( e_stop );
	cudaEventDestroy( e_start );

	cudaFree( test_sort_keys );
	cudaFree( test_sort_values );
	cudaFree( test_sort_indices );
	cudaFreeHost( test_sort_keys_host );
	cudaFreeHost( test_sort_values_host );
	cudaFreeHost( test_sort_indices_host );

	return 0;
}
