
// Code original : MX^ADD (mxadd@mxadd.org)

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define BITONIC_SORT_THREAD_LIMIT 64

#define BITONIC_BIN_CAPACITY 128
#define BITONIC_BUCKET_SIZE 256

#define BITONIC_INIT_THREAD_LIMIT 256
#define BITONIC_FINAL_THREAD_LIMIT 256

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
template <typename key_type>
__global__ void kernel_bitonicSort( key_type* keys, uint* indices, uint length )
{
	__shared__ key_type s_keys[BITONIC_BIN_CAPACITY];
	__shared__ uint s_indices[BITONIC_BIN_CAPACITY];

	const uint n = threadIdx.x;
	const uint o = blockIdx.x * blockDim.x;

	const uint size = min( BITONIC_BIN_CAPACITY, length - o );

	if ( size <= 1 ) return;

	// chaque thread copie un élément 
	if ( n < size )
	{
		s_keys[n] = keys[o+n];
		s_indices[n] = indices[o+n];
	}

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
					key_type ka = s_keys[n];
					key_type kb = s_keys[ixj];

					uint ia = s_indices[n];
					uint ib = s_indices[ixj];

					if ( (n & i) == 0 )
					{
						if ( ka > kb )
						{
							s_keys[n] = kb;
							s_keys[ixj] = ka;

							s_indices[n] = ib;
							s_indices[ixj] = ia;
						}
					}
					else
					{
						if ( ka < kb )
						{
							s_keys[n] = kb;
							s_keys[ixj] = ka;

							s_indices[n] = ib;
							s_indices[ixj] = ia;
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

		key_type ka, kb;
		uint ia, ib;

		for ( uint i = 0 ; i < j ; i++ )
		{
			if ( ( i & 0x01 ) == 0 )
			{
				// [0|1] [2|3] [4|5] [6|7] ...
				if ( ( k + 1 ) < size )
				{
					ka = s_keys[k+0];
					kb = s_keys[k+1];

					ia = s_indices[k+0];
					ib = s_indices[k+1];

					if ( ka > kb )
					{
						s_keys[k+0] = kb;
						s_keys[k+1] = ka;

						s_indices[k+0] = ib;
						s_indices[k+1] = ia;
					}
				}
			}
			else
			{
				// [1|2] [3|4] [5|6] [7|8] ...
				if ( ( k + 2 ) < size )
				{
					ka = s_keys[k+1];
					kb = s_keys[k+2];

					ia = s_indices[k+1];
					ib = s_indices[k+2];

					if ( ka > kb )
					{
						s_keys[k+1] = kb;
						s_keys[k+2] = ka;

						s_indices[k+1] = ib;
						s_indices[k+2] = ia;
					}
				}
			}

			__syncthreads();
		}
	}

	if ( n < size )
	{
		keys[o+n] = s_keys[n];
		indices[o+n] = s_indices[n];
	}
}

// mélange deux à deux chacun des bins en conservant les éléments triés
template <typename key_type>
__global__ void kernel_bitonicMerge2x( key_type* keys, uint* indices, uint length )
{
	__shared__ key_type s_keys_in[BITONIC_BUCKET_SIZE];
	__shared__ key_type s_keys_out[BITONIC_BUCKET_SIZE];

	__shared__ uint s_indices_in[BITONIC_BUCKET_SIZE];
	__shared__ uint s_indices_out[BITONIC_BUCKET_SIZE];

	uint o = ( blockIdx.x * blockDim.x ) << 1;

	uint k = blockDim.x;
	uint n = threadIdx.x;

	uint p = k; // fin de la première liste
	uint q = min( p, length - o - p ) + p; // fin de la seconde liste

	uint i = n; // début de la première liste pour ce thread
	uint j = p + n; // début de la seconde liste pour ce thread

	// chaque thread copie un élément dans chacune des deux listes
	{
		s_keys_in[i] = keys[o+i];
		s_indices_in[i] = indices[o+i];

		if ( j < q )
		{
			s_keys_in[j] = keys[o+j];
			s_indices_in[j] = indices[o+j];
		}
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
			if ( s_keys_in[a] <= s_keys_in[b] )
			{
				s_keys_out[c] = s_keys_in[a];
				s_indices_out[c] = s_indices_in[a];
				c++; a++;
			}
			else
			{
				s_keys_out[c] = s_keys_in[b];
				s_indices_out[c] = s_indices_in[b];
				c++; b++;
			}
		}

		// on oublie pas de recopier la queue

		while ( a < p )
		{
			s_keys_out[c] = s_keys_in[a];
			s_indices_out[c] = s_indices_in[a];
			c++; a++;
		}

		while ( b < q )
		{
			s_keys_out[c] = s_keys_in[b];
			s_indices_out[c] = s_indices_in[b];
			c++; b++;
		}
	}

	__syncthreads();

	// enfin chaque thread recopie un élément depuis chacune des deux listes
	{
		keys[o+i] = s_keys_out[i];
		indices[o+i] = s_indices_out[i];

		if ( j < p + q )
		{
			keys[o+j] = s_keys_out[j];
			indices[o+j] = s_indices_out[j];
		}
	}
}

template <typename key_type>
__global__ void kernel_bitonicMergeAny( key_type* keys, key_type* keys_buffer, uint* indices, uint* indices_buffer, uint length, uint bucket_size )
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
			keys_buffer[a] = keys[a];
			indices_buffer[a] = indices[a];
			a++;
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
			key_type ku = keys[a];
			key_type kv = keys[b];

			uint iu = indices[a];
			uint iv = indices[b];

			if ( ku <= kv )
			{
				keys_buffer[c] = ku;
				indices_buffer[c] = iu;
				a++; c++;
			}
			else
			{
				keys_buffer[c] = kv;
				indices_buffer[c] = iv;
				b++; c++;
			}
		}

		// ... et on recopie la queue

		while ( a < p )
		{
			keys_buffer[c] = keys[a];
			indices_buffer[c] = indices[a];
			a++; c++;
		}

		while ( b < q )
		{
			keys_buffer[c] = keys[b];
			indices_buffer[c] = indices[b];
			b++; c++;
		}
	}
}

template <typename key_type>
__global__ void kernel_bitonicMergeAnyParallel( key_type* keys, key_type* keys_buffer, uint* indices, uint* indices_buffer, uint length, uint bucket_size )
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

			keys_buffer[a+n] = keys[a+n];
			indices_buffer[a+n] = indices[a+n];

			a += BITONIC_SORT_THREAD_LIMIT;
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

		__shared__ key_type s_keys0[BITONIC_SORT_THREAD_LIMIT];
		__shared__ key_type s_keys1[BITONIC_SORT_THREAD_LIMIT];
		__shared__ uint s_indices0[BITONIC_SORT_THREAD_LIMIT];
		__shared__ uint s_indices1[BITONIC_SORT_THREAD_LIMIT];

		uint cnt0 = 0;
		uint cnt1 = 0;

		while ( a < p && b < q )
		{
			if ( cnt0 == 0 )
			{
				cnt0 = a + n;

				if ( cnt0 < p )
				{
					s_keys0[n] = keys[cnt0];
					s_indices0[n] = indices[cnt0];
				}

				cnt0 = BITONIC_SORT_THREAD_LIMIT;
			}

			if ( cnt1 == 0 )
			{
				cnt1 = b + n;

				if ( cnt1 < q )
				{
					s_keys1[n] = keys[cnt1];
					s_indices1[n] = indices[cnt1];
				}

				cnt1 = BITONIC_SORT_THREAD_LIMIT;
			}

			__syncthreads();

			if ( n == 0 )
			{
				while ( a < p && b < q )
				{
					key_type key0 = s_keys0[BITONIC_SORT_THREAD_LIMIT-cnt0];
					key_type key1 = s_keys1[BITONIC_SORT_THREAD_LIMIT-cnt1];
					uint idx0 = s_indices0[BITONIC_SORT_THREAD_LIMIT-cnt0];
					uint idx1 = s_indices1[BITONIC_SORT_THREAD_LIMIT-cnt1];

					if ( key0 <= key1 )
					{
						keys_buffer[c] = key0;
						indices_buffer[c] = idx0;

						c++;
						a++;
						cnt0--;

						if ( cnt0 == 0 ) break;
					}
					else
					{
						keys_buffer[c]  = key1;
						indices_buffer[c]  = idx1;

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

			keys_buffer[c+n] = keys[a+n];
			indices_buffer[c+n] = indices[a+n];

			c += BITONIC_SORT_THREAD_LIMIT;
			a += BITONIC_SORT_THREAD_LIMIT;
		}

		while ( b < q )
		{
			if ( ( b + n ) >= q ) break;

			keys_buffer[c+n] = keys[b+n];
			indices_buffer[c+n] = indices[b+n];

			c += BITONIC_SORT_THREAD_LIMIT;
			b += BITONIC_SORT_THREAD_LIMIT;
		}
	}
}

template <typename value_type>
__global__ void kernel_bitonicSortFinal( value_type* values, value_type* buffer, uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
		values[n] = buffer[indices[n]];
}

__global__ void kernel_bitonicSortInit( uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
		indices[n] = n;
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

template <typename key_type, typename value_type>
void CUDA_bitonicSortGeneral( key_type* keys, value_type* values, uint length )
{
	if ( length <= 1 ) return;

	uint* indices; cudaMalloc( (void**)&indices, length * sizeof(uint) );

	// on initialise les indices
	{
		dim3 db = dim3( BITONIC_INIT_THREAD_LIMIT );
		dim3 dg = dim3( ( length + db.x - 1 ) / db.x );

		kernel_bitonicSortInit<<<dg,db>>>
		(
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );
	}

	const uint bin_count = ( length + ( BITONIC_BIN_CAPACITY - 1 ) ) / BITONIC_BIN_CAPACITY;
	const uint thread_count = min( length, BITONIC_BIN_CAPACITY );

	// on trie les éléments au sein de chaque bin
	{
		dim3 db = dim3( thread_count );
		dim3 dg = dim3( bin_count );

		kernel_bitonicSort<key_type><<<dg,db>>>
		(
			keys,
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );
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

			kernel_bitonicMerge2x<key_type><<<dg,db>>>
			(
				keys,
				indices,
				length
			);

			_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
						"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );
		}

		// si le tableau a moins d'éléments que BITONIC_BUCKET_SIZE
		// alors on a fini sinon on mélange à nouveau les buckets restant
		if ( length > BITONIC_BUCKET_SIZE )
		{
			key_type* keys_buffer;
			uint* indices_buffer;

			cudaMalloc( (void**)&keys_buffer, length * sizeof(key_type) );
			cudaMalloc( (void**)&indices_buffer, length * sizeof(uint) );

			key_type* keys_src = keys;
			key_type* keys_dst = keys_buffer;

			uint* indices_src = indices;
			uint* indices_dst = indices_buffer;

			uint bucket_size = BITONIC_BUCKET_SIZE;

			while ( bucket_size < length )
			{
				const uint thread_required = ( ( length + ( ( bucket_size * 2 ) - 1 ) ) / ( bucket_size * 2 ) );

				const uint grid_dim = ( thread_required + ( BITONIC_SORT_THREAD_LIMIT - 1 ) ) / BITONIC_SORT_THREAD_LIMIT;
				const uint block_dim = min( thread_required, BITONIC_SORT_THREAD_LIMIT );

				if ( grid_dim > 1 )
				{
					dim3 db = dim3( block_dim );
					dim3 dg = dim3( grid_dim );

					kernel_bitonicMergeAny<key_type><<<dg,db>>>
					(
						keys_src,
						keys_dst,
						indices_src,
						indices_dst,
						length,
						bucket_size
					);

					_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
								"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );
				}
				else
				{
					dim3 db = dim3( BITONIC_SORT_THREAD_LIMIT );
					dim3 dg = dim3( grid_dim * block_dim );

					kernel_bitonicMergeAnyParallel<key_type><<<dg,db>>>
					(
						keys_src,
						keys_dst,
						indices_src,
						indices_dst,
						length,
						bucket_size
					);

					_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
								"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );
				}

				// on échange les pointeurs source et destination
				{
					key_type* keys_tmp = keys_src;
					keys_src = keys_dst;
					keys_dst = keys_tmp;

					uint* indices_tmp = indices_src;
					indices_src = indices_dst;
					indices_dst = indices_tmp;
				}

				bucket_size *= 2;
			}

			if ( keys_src != keys )
			{
				#ifdef __DEVICE_EMULATION__
					cudaMemcpy( keys, keys_src, length * sizeof(key_type), cudaMemcpyHostToHost );
				#else
					cudaMemcpy( keys, keys_src, length * sizeof(key_type), cudaMemcpyDeviceToDevice );
				#endif
			}

			if ( indices_src != indices )
			{
				#ifdef __DEVICE_EMULATION__
					cudaMemcpy( indices, indices_src, length * sizeof(uint), cudaMemcpyHostToHost );
				#else
					cudaMemcpy( indices, indices_src, length * sizeof(uint), cudaMemcpyDeviceToDevice );
				#endif
			}

			cudaFree( indices_buffer );
			cudaFree( keys_buffer );
		}
	}

	// on effectue enfin le tri des valeurs en fonction des indices
	{
		value_type* buffer;

		cudaMalloc( (void**)&buffer, length * sizeof(value_type) );
		cudaMemcpy( buffer, values, length * sizeof(value_type), cudaMemcpyDeviceToDevice );

		dim3 db = dim3( BITONIC_FINAL_THREAD_LIMIT );
		dim3 dg = dim3( ( length + db.x - 1 ) / db.x );

		kernel_bitonicSortFinal<value_type><<<dg,db>>>
		(
			values,
			buffer,
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_bitonicSortGeneral() : Unable to execute CUDA kernel." );

		cudaFree( buffer );
	}

	cudaFree( indices );
}

// force nvcc à générer le code pour chacun des noyaux
// ajouter les lignes correspondant aux types nécessaires

void dummy_bitonicSortGeneral()
{
 	CUDA_bitonicSortGeneral<uint,uint>( NULL, NULL, 0 );
}
