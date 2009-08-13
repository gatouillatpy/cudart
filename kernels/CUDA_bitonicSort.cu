
// Code original : MX^ADD (mxadd@mxadd.org)

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BITONIC_THREAD_LIMIT 64

#define BITONIC_BIN_CAPACITY 128
#define BITONIC_BUCKET_SIZE 256

/***********************************************************************************/
/** MACROS GPU                                                                    **/
/***********************************************************************************/

inline __device__ void swap( uint& a, uint& b )
{
	a ^= b;
	b ^= a;
	a ^= b;
}

/***********************************************************************************/
/** NOYAUX                                                                        **/
/***********************************************************************************/

// triage des BITONIC_BIN_CAPACITY éléments au sein de chacun des bins
template <typename value_type>
__global__ void kernel_bitonicSort( value_type* values, uint length )
{
	__shared__ value_type s_values[BITONIC_BIN_CAPACITY];

	const uint n = threadIdx.x;
	const uint o = blockIdx.x * blockDim.x;

	const uint size = min( BITONIC_BIN_CAPACITY, length - o );

	if ( size <= 1 ) return;

	// chaque thread copie un élément 
	if ( n < size ) s_values[n] = values[o+n];

	__syncthreads();

	// si ce block traite un bin complet alors
	// on procède à un tri bitonic du bin
	if ( size == BITONIC_BIN_CAPACITY )
	{
		for ( uint i = 2 ; i <= BITONIC_BIN_CAPACITY ; i <<= 1 )
		{
			for ( uint j = (i >> 1) ; j > 0 ; j >>= 1 )
			{
				uint ixj = n ^ j;

				if ( ixj > n )
				{
					value_type a = s_values[n];
					value_type b = s_values[ixj];

					if ( (n & i) == 0 )
					{
						if ( a > b )
						{
							s_values[n] = b;
							s_values[ixj] = a;
						}
					}
					else
					{
						if ( a < b )
						{
							s_values[n] = b;
							s_values[ixj] = a;
						}
					}
				}

				__syncthreads();
			}
		}
	}
	// sinon on effectue un tri à bulles du bin
	else
	{
		const uint k = (n << 1);
		const uint j = (size << 1);

		value_type a, b;

		for ( uint i = 0 ; i < j ; i++ )
		{
			if ( ( i & 0x01 ) == 0 )
			{
				// [0|1] [2|3] [4|5] [6|7] ...
				if ( ( k + 1 ) < size )
				{
					a = s_values[k+0];
					b = s_values[k+1];

					if ( a > b )
					{
						s_values[k+0] = b;
						s_values[k+1] = a;
					}
				}
			}
			else
			{
				// [1|2] [3|4] [5|6] [7|8] ...
				if ( ( k + 2 ) < size )
				{
					a = s_values[k+1];
					b = s_values[k+2];

					if ( a > b )
					{
						s_values[k+1] = b;
						s_values[k+2] = a;
					}
				}
			}

			__syncthreads();
		}
	}

	if ( n < size ) values[o+n] = s_values[n];
}

// mélange deux à deux chacun des bins en conservant les éléments triés
template <typename value_type>
__global__ void kernel_bitonicMerge2x( value_type* values, uint length )
{
	__shared__ value_type s_values_in[BITONIC_BUCKET_SIZE];
	__shared__ value_type s_values_out[BITONIC_BUCKET_SIZE];

	uint o = ( blockIdx.x * blockDim.x ) << 1;

	uint k = blockDim.x;
	uint n = threadIdx.x;

	uint p = k; // fin de la première liste
	uint q = min( p, length - o - p ) + p; // fin de la seconde liste

	uint i = n; // début de la première liste pour ce thread
	uint j = p + n; // début de la seconde liste pour ce thread

	// chaque thread copie un élément dans chacune des deux listes
	{
		s_values_in[i] = values[o+i];

		if ( j < q )
			s_values_in[j] = values[o+j];
	}

	__syncthreads();

	// le premier thread mélange ensuite les deux listes
	// todo : voir si on ne peut pas paralléliser cela
	if ( n == 0 )
	{
		uint a = 0;
		uint b = p;
		uint c = 0;

		while ( a < p && b < q )
		{
			if ( s_values_in[a] <= s_values_in[b] )
				s_values_out[c++] = s_values_in[a++];
			else
				s_values_out[c++] = s_values_in[b++];
		}

		// on oublie pas de recopier la queue

		while ( a < p )
			s_values_out[c++] = s_values_in[a++];

		while ( b < q )
			s_values_out[c++] = s_values_in[b++];
	}

	__syncthreads();

	// enfin chaque thread recopie un élément depuis chacune des deux listes
	{
		values[o+i] = s_values_out[i];

		if ( j < p + q )
			values[o+j] = s_values_out[j];
	}
}

template <typename value_type>
__global__ void kernel_bitonicMergeAny( value_type* values, value_type* buffer, uint length, uint bucket_size )
{
	const uint o = blockIdx.x * blockDim.x + threadIdx.x;

	const uint i = o * bucket_size * 2;
	const uint j = o * bucket_size * 2 + bucket_size;

	const uint p = min( length, i + bucket_size );
	const uint q = min( length, j + bucket_size );

	// s'il s'agit de la queue de la liste alors on n'a qu'à recopier
	if ( j >= length )
	{
		unsigned int a = i;

		while ( a < length )
		{
			buffer[a] = values[a]; a++;
		}
	}
	// sinon on mélange ...
	else
	{
		uint a = i;
		uint b = j;
		uint c = i;

		while ( a < p && b < q )
		{
			value_type u = values[a];
			value_type v = values[b];

			if ( u <= v )
			{
				buffer[c++] = u;
				a++;
			}
			else
			{
				buffer[c++] = v;
				b++;
			}
		}

		// ... et on recopie la queue

		while ( a < p )
		{
			buffer[c++] = values[a];
			a++;
		}

		while ( b < q )
		{
			buffer[c++] = values[b];
			b++;
		}
	}
}

template <typename value_type>
__global__ void kernel_bitonicMergeAnyParallel( value_type* values, value_type* buffer, uint length, uint bucket_size )
{
	const uint n = threadIdx.x;
	const uint o = blockIdx.x;

	uint a = o * bucket_size * 2;
	uint b = o * bucket_size * 2 + bucket_size;

	const uint p = min( length, a + bucket_size );
	const uint q = min( length, b + bucket_size );

	// s'il s'agit de la queue de la liste alors on n'a qu'à recopier
	if ( b >= length )
	{
		while ( a < length )
		{
			if ( a + n >= length ) break;

			buffer[a+n] = values[a+n];

			a += BITONIC_THREAD_LIMIT;
		}
	}
	// sinon on mélange ...
	else
	{
		uint c = a;

		__shared__ uint res_id;
		__shared__ uint id0;
		__shared__ uint id1;
		__shared__ uint cid0;
		__shared__ uint cid1;

		__shared__ value_type s_values0[BITONIC_THREAD_LIMIT];
		__shared__ value_type s_values1[BITONIC_THREAD_LIMIT];

		uint cnt0 = 0;
		uint cnt1 = 0;

		while ( a < p && b < q )
		{
			if ( cnt0 == 0 )
			{
				cnt0 = a + n;

				if ( cnt0 < p )
					s_values0[n] = values[cnt0];

				cnt0 = BITONIC_THREAD_LIMIT;
			}

			if ( cnt1 == 0 )
			{
				cnt1 = b + n;

				if ( cnt1 < q )
					s_values1[n] = values[cnt1];

				cnt1 = BITONIC_THREAD_LIMIT;
			}

			__syncthreads();

			if ( n == 0 )
			{
				while ( a < p && b < q )
				{
					value_type val0 = s_values0[BITONIC_THREAD_LIMIT-cnt0];
					value_type val1 = s_values1[BITONIC_THREAD_LIMIT-cnt1];

					if ( val0 <= val1 )
					{
						buffer[c]  = val0;

						c++;
						a++;
						cnt0--;

						if ( cnt0 == 0 ) break;
					}
					else
					{
						buffer[c]  = val1;

						c++;
						b++;
						cnt1--;

						if ( cnt1 == 0 ) break;
					}
				}

				id0    = a;
				id1    = b;
				res_id = c;

				cid0 = cnt0;
				cid1 = cnt1;
			}

			__syncthreads();

			a = id0;
			b = id1;
			c = res_id;

			cnt0 = cid0;
			cnt1 = cid1;
		}

		// ... et on recopie la queue

		while ( a < p )
		{
			if ( ( a + n ) >= p ) break;

			buffer[c+n] = values[a+n];

			c += BITONIC_THREAD_LIMIT;
			a += BITONIC_THREAD_LIMIT;
		}

		while ( b < q )
		{
			if ( ( b + n ) >= q ) break;

			buffer[c+n] = values[b+n];

			c += BITONIC_THREAD_LIMIT;
			b += BITONIC_THREAD_LIMIT;
		}
	}
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

template <typename value_type>
void CUDA_bitonicSort( value_type* values, uint length )
{
	if ( length <= 1 ) return;

	const uint bin_count = ( length + ( BITONIC_BIN_CAPACITY - 1 ) ) / BITONIC_BIN_CAPACITY;
	const uint thread_count = min( length, BITONIC_BIN_CAPACITY );

	// on trie les éléments au sein de chaque bin
	{
		dim3 db = dim3( thread_count );
		dim3 dg = dim3( bin_count );

		kernel_bitonicSort<value_type><<<dg,db>>>
		(
			values,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bitonicSort() : Unable to execute CUDA kernel." );
	}

	// si le tableau a moins d'éléments que BITONIC_BIN_CAPACITY alors on
	// a fini sinon on mélange les bins en utilisant la mémoire partagée
	if ( length > BITONIC_BIN_CAPACITY )
	{
		// on mélange d'abord chaque bin deux à deux
		// nb : le dernier bin n'est pas mélangé si bin_count est impair
		{
			dim3 db = dim3( BITONIC_BIN_CAPACITY );
			dim3 dg = dim3( bin_count / 2 );

			kernel_bitonicMerge2x<value_type><<<dg,db>>>
			(
				values,
				length
			);

			_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
						"CUDA_bitonicSort() : Unable to execute CUDA kernel." );
		}

		// si le tableau a moins d'éléments que BITONIC_BUCKET_SIZE
		// alors on a fini sinon on mélange à nouveau les buckets restant
		if ( length > BITONIC_BUCKET_SIZE )
		{
			value_type* buffer;

			cudaMalloc( (void**)&buffer, length * sizeof(value_type) );

			value_type* src_ptr = values;
			value_type* dst_ptr = buffer;

			uint bucket_size = BITONIC_BUCKET_SIZE;

			while ( bucket_size < length )
			{
				const uint thread_required = ( ( length + ( ( bucket_size * 2 ) - 1 ) ) / ( bucket_size * 2 ) );

				const uint grid_dim = ( thread_required + ( BITONIC_THREAD_LIMIT - 1 ) ) / BITONIC_THREAD_LIMIT;
				const uint block_dim = min( thread_required, BITONIC_THREAD_LIMIT );

				if ( grid_dim > 1 )
				{
					dim3 db = dim3( block_dim );
					dim3 dg = dim3( grid_dim );

					kernel_bitonicMergeAny<value_type><<<dg,db>>>
					(
						src_ptr,
						dst_ptr,
						length,
						bucket_size
					);

					_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
								"CUDA_bitonicSort() : Unable to execute CUDA kernel." );
				}
				else
				{
					dim3 db = dim3( BITONIC_THREAD_LIMIT );
					dim3 dg = dim3( grid_dim * block_dim );

					kernel_bitonicMergeAnyParallel<value_type><<<dg,db>>>
					(
						src_ptr,
						dst_ptr,
						length,
						bucket_size
					);

					_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
								"CUDA_bitonicSort() : Unable to execute CUDA kernel." );
				}

				// on échange src_ptr et dst_ptr
				value_type* tmp_ptr = src_ptr;
				src_ptr = dst_ptr;
				dst_ptr = tmp_ptr;

				bucket_size *= 2;
			}

			if ( src_ptr != values )
			{
				#ifdef __DEVICE_EMULATION__
					cudaMemcpy( values, src_ptr, length * sizeof(value_type), cudaMemcpyHostToHost );
				#else
					cudaMemcpy( values, src_ptr, length * sizeof(value_type), cudaMemcpyDeviceToDevice );
				#endif
			}

			cudaFree( buffer );
		}
	}
}

// force nvcc à générer le code pour chacun des noyaux
// ajouter les lignes correspondant aux types nécessaires

void dummy_bitonicSort()
{
// 	CUDA_bitonicSort<char>( NULL, 0 );
// 	CUDA_bitonicSort<short>( NULL, 0 );
// 	CUDA_bitonicSort<int>( NULL, 0 );
// 	CUDA_bitonicSort<long>( NULL, 0 );
// 
// 	CUDA_bitonicSort<unsigned char>( NULL, 0 );
// 	CUDA_bitonicSort<unsigned short>( NULL, 0 );
 	CUDA_bitonicSort<unsigned int>( NULL, 0 );
// 	CUDA_bitonicSort<unsigned long>( NULL, 0 );
// 
// 	CUDA_bitonicSort<float>( NULL, 0 );
// 	CUDA_bitonicSort<double>( NULL, 0 );
}
