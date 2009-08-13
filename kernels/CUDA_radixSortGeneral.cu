
// Code original : MX^ADD (mxadd@mxadd.org)

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "../CUDACommon.h"

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#define RADIX_THREAD_COUNT 256
#define RADIX_THREAD_CAPACITY 32

/***********************************************************************************/
/** TEXTURES                                                                      **/
/***********************************************************************************/

texture <uint2, 1, cudaReadModeElementType> tex_pairs;

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void CUDA_bubbleSort512( uint* keys, uint* indices, uint length, bool init_indices = false );

template <typename key_type, typename value_type>
void CUDA_bitonicSortGeneral( value_type* keys, value_type* values, uint length );

/***********************************************************************************/
/** NOYAUX                                                                        **/
/***********************************************************************************/

__global__ static void kernel_radixFillHistogram
(
	uint* hist_out, 
	const uint length, 
	const uint thread_count,
	const uint thread_capa,
	const uint hist_offset_array,
	const uint hist_offset,
	const uint bit_shift
)
{
	__shared__ uint s_histogram[256];

	const uint n = threadIdx.x;
	const uint k = thread_count * thread_capa;
	const uint o = blockIdx.x * k;

	s_histogram[n] = 0;

	__syncthreads();

	for ( uint i = 0 ; i < k ; i += thread_count )
	{
		const uint idx = o + i + n;

		#ifdef CUDA_ATOMIC_ON_SHARED_MEMORY

			if ( idx >= length ) break;

			uint2 t_pair = tex1Dfetch( tex_pairs, idx + hist_offset_array );

			atomicAdd( &s_histogram[(t_pair.x >> bit_shift) & 0xFF], 1 );

		#else

			uint2 t_pair;

			bool pwnt = false;

			if ( idx < length )
			{
				t_pair = tex1Dfetch( tex_pairs, idx + hist_offset_array );

				pwnt = true;
			}

			for ( uint j = 0 ; j < thread_count ; j++ )
			{
				if ( j == n && pwnt == true )
					s_histogram[(t_pair.x >>  bit_shift) & 0xFF]++;

				__syncthreads();
			}

		#endif
	}

	__syncthreads();

	atomicAdd( &hist_out[hist_offset+n], s_histogram[n] );
}

__global__ static void kernel_radixSumHistogram
(
	uint* histogram, 
	const uint block_count
)
{
	const uint n = threadIdx.x;

	uint sum = 0;

	for ( uint i = 0 ; i < block_count ; i++ )
		sum += histogram[(i + 1) * 256 + n];

	histogram[n] = sum;
}

__global__ static void kernel_radixCalcCounters
(
	uint* histogram, 
	const uint block_count
)
{
	const uint n = threadIdx.x;

	__shared__ uint s_histogram[256];

	s_histogram[n] = histogram[n];

	uint counters = 0;

	__syncthreads();

	for ( uint i = 0 ; i < n ; i++ ) counters += s_histogram[i];

	histogram[n] = counters;

	if ( block_count > 1 )
	{
		for ( uint i = 1 ; i < block_count ; i++ )
		{
			int k = i * 256 + n;

			counters += histogram[k];

			histogram[k] = counters;
		}
	}
}

__global__ static void kernel_radixSortData
(
	uint2* pairs, 
	uint* counters,
	const uint length, 
	const uint thread_capa,
	const uint bit_shift
)
{
	__shared__ uint s_counters[256];

	const uint n = threadIdx.x;
	const uint k = thread_capa << 8;
	const uint o = blockIdx.x * k;

	uint place; uint2 t_pair;

	s_counters[n] = counters[(blockIdx.x << 8) + n];

	__syncthreads();

	for ( uint i = 0 ; i < k ; i+= RADIX_THREAD_COUNT )
	{
		const uint idx = o + i + n;

		if ( idx < length )
		{
			t_pair = tex1Dfetch( tex_pairs, idx );
			
			place = (t_pair.x >> bit_shift) & 0xFF;
		}
		else
		{
			place = 0xFFFFFFFF;
		}

		#define ATT_ADD_OFF(_arg) case _arg: if ( place != 0xFFFFFFFF ) pairs[s_counters[place]++] = t_pair; break;

		#pragma unroll
		for ( uint t = 0 ; t < RADIX_THREAD_COUNT ; t += warpSize )
		{
			if ( (n - t) <= 0x1F )
			{
				switch ( n & 0x1F )
				{
					ATT_ADD_OFF( 0);
					ATT_ADD_OFF( 1);
					ATT_ADD_OFF( 2);
					ATT_ADD_OFF( 3);
					ATT_ADD_OFF( 4);
					ATT_ADD_OFF( 5);
					ATT_ADD_OFF( 6);
					ATT_ADD_OFF( 7);

					ATT_ADD_OFF( 8);
					ATT_ADD_OFF( 9);
					ATT_ADD_OFF(10);
					ATT_ADD_OFF(11);
					ATT_ADD_OFF(12);
					ATT_ADD_OFF(13);
					ATT_ADD_OFF(14);
					ATT_ADD_OFF(15);

					ATT_ADD_OFF(16);
					ATT_ADD_OFF(17);
					ATT_ADD_OFF(18);
					ATT_ADD_OFF(19);
					ATT_ADD_OFF(20);
					ATT_ADD_OFF(21);
					ATT_ADD_OFF(22);
					ATT_ADD_OFF(23);

					ATT_ADD_OFF(24);
					ATT_ADD_OFF(25);
					ATT_ADD_OFF(26);
					ATT_ADD_OFF(27);
					ATT_ADD_OFF(28);
					ATT_ADD_OFF(29);
					ATT_ADD_OFF(30);
					ATT_ADD_OFF(31);
				}
			}

			__syncthreads();		 
		}

		#undef ATT_ADD_OFF
	}
}

__global__ static void kernel_radixSetPairs( uint2* pairs, uint* keys, uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
	{
		uint2 t_pair;

		t_pair.x = keys[n];
		t_pair.y = indices[n];

		pairs[n] = t_pair;
	}
}

__global__ static void kernel_radixGetPairs( uint2* pairs, uint* keys, uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
	{
		uint2 t_pair = pairs[n];

		keys[n] = t_pair.x;
		indices[n] = t_pair.y;
	}
}

__global__ static void kernel_radixSortInit( uint* indices, uint length )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < length )
		indices[n] = n;
}

/***********************************************************************************/
/** FONCTION                                                                      **/
/***********************************************************************************/

static void radixStep
(
	uint2* src_ptr, 
	uint2* dst_ptr, 
	uint length,
	uint* histogram,
	uint bit_shift,
	uint thread_capa,
	uint block_count
)
{
	uint hist_thread_count, hist_thread_capa, hist_block_count;

	cudaMemset( histogram, 0, ( block_count + 1 ) * 256 * sizeof(uint) );

	// on monte la texture pour les données en entrée
	{
		cudaChannelFormatDesc desc = { 32, 32, 0, 0, cudaChannelFormatKindUnsigned };

		if ( cudaBindTexture( NULL, tex_pairs, src_ptr, desc ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDA_radixSort() : Unable to bind CUDA texture." );
	}

	const uint overall_capa = thread_capa * RADIX_THREAD_COUNT;

	// on remplit chacune des tranches de l'histogramme
	for ( uint i = 0 ; i < block_count ; i++ )
	{
		const uint required_capa = min( overall_capa, length - i * overall_capa );

		hist_thread_count = 256;
		hist_thread_capa = 8;
		hist_block_count = ( required_capa + hist_thread_count * hist_thread_capa - 1 ) / ( hist_thread_count * hist_thread_capa );

		uint hist_out_offset;

		if ( block_count == 1 )
			hist_out_offset = 0;
		else
			hist_out_offset = ( i + 1 ) * 256;

		dim3 db = dim3( hist_thread_count );
		dim3 dg = dim3( hist_block_count );

		kernel_radixFillHistogram<<<dg,db>>>
		(
			histogram,
			required_capa,
			hist_thread_count,
			hist_thread_capa,
			i * overall_capa,
			hist_out_offset,
			bit_shift
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSort() : Unable to execute CUDA kernel." );
	}

	// si l'histogramme possède plus qu'une
	// tranche alors on en fait la somme
	if ( block_count > 1 )
	{
		dim3 db = dim3( 256 );
		dim3 dg = dim3( 1 );

		kernel_radixSumHistogram<<<dg,db>>>
		(
			histogram,
			block_count
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSort() : Unable to execute CUDA kernel." );
	}

	// on calcule les compteurs
	{
		dim3 db = dim3( 256 );
		dim3 dg = dim3( 1 );

		kernel_radixCalcCounters<<<dg,db>>>
		(
			histogram,
			block_count
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSort() : Unable to execute CUDA kernel." );
	}

	// enfin on trie
	{
		dim3 db = dim3( RADIX_THREAD_COUNT );
		dim3 dg = dim3( block_count );

		kernel_radixSortData<<<dg,db>>>
		(
			dst_ptr,
			histogram,
			length,
			thread_capa,
			bit_shift
		);

		cudaThreadSynchronize();

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSort() : Unable to execute CUDA kernel." );
	}

	cudaUnbindTexture( tex_pairs );
}

void CUDA_radixSortGeneral( uint* keys, uint* indices, uint length, uint bit_depth = 32, bool init_indices = false )
{
	uint byte_depth = 1 + (bit_depth - 1) / 8;

	if ( length <= 512 )
	{
		CUDA_bubbleSort512( keys, indices, length, init_indices );

		return;
	}

	// on initialise les indices
	if ( init_indices )
	{
		dim3 db = dim3( RADIX_THREAD_COUNT );
		dim3 dg = dim3( ( length + db.x - 1 ) / db.x );

		kernel_radixSortInit<<<dg,db>>>
		(
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSortGeneral() : Unable to execute CUDA kernel." );
	}

	if ( length <= byte_depth * RADIX_THREAD_COUNT * RADIX_THREAD_CAPACITY )
	{
		CUDA_bitonicSortGeneral<uint,uint>( keys, indices, length );

		return;
	}

	// on effectue le tri
	{
		uint block_count = ( length + ( RADIX_THREAD_COUNT * RADIX_THREAD_CAPACITY - 1 ) ) / ( RADIX_THREAD_COUNT * RADIX_THREAD_CAPACITY );

		uint* histogram; uint2* pairs; uint2* buffer;

		cudaMalloc( (void**)&histogram, ( block_count + 1 ) * 256 * sizeof(uint) );
		cudaMalloc( (void**)&pairs, length * sizeof(uint2) );
		cudaMalloc( (void**)&buffer, length * sizeof(uint2) );

		dim3 db = dim3( RADIX_THREAD_COUNT );
		dim3 dg = dim3( ( length + db.x - 1 ) / db.x );

		kernel_radixSetPairs<<<dg,db>>>
		(
			pairs,
			keys,
			indices,
			length
		);

		_assert( cudaGetLastError() == cudaSuccess, __FILE__, __LINE__,
					"CUDA_radixSortGeneral() : Unable to execute CUDA kernel." );

		if ( byte_depth > 0 )
			radixStep( pairs, buffer, length, histogram,  0, RADIX_THREAD_CAPACITY, block_count );

		if ( byte_depth > 1 )
			radixStep( buffer, pairs, length, histogram,  8, RADIX_THREAD_CAPACITY, block_count );

		if ( byte_depth > 2 )
			radixStep( pairs, buffer, length, histogram, 16, RADIX_THREAD_CAPACITY, block_count );

		if ( byte_depth > 3 )
			radixStep( buffer, pairs, length, histogram, 24, RADIX_THREAD_CAPACITY, block_count );

		if ( byte_depth & 0x01 )
			cudaMemcpy( pairs, buffer, length * sizeof(uint2), cudaMemcpyDeviceToDevice );

		kernel_radixGetPairs<<<dg,db>>>
		(
			pairs,
			keys,
			indices,
			length
		);

		cudaFree( pairs );
		cudaFree( buffer );
		cudaFree( histogram );
	}
}
