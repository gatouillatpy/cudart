
#ifndef _CUDA_COMMON
#define _CUDA_COMMON

#pragma warning (disable : 4251)

/***********************************************************************************/
/** DEFINITIONS                                                                   **/
/***********************************************************************************/

#ifndef _WIN32_WINNT
	#define _WIN32_WINNT 0x400
#endif

#define DIRECTINPUT_VERSION  0x0800

/***********************************************************************************/
/** TYPES                                                                         **/
/***********************************************************************************/

typedef unsigned long long qword;
typedef unsigned int dword;
typedef unsigned short word;
typedef long long wide;
typedef unsigned long long uwide;
typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned char byte;

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include <assert.h>

#include "../../common/mat44.h"

#include "types/_cuda_char.h"
#include "types/_cuda_char1.h"
#include "types/_cuda_char2.h"
#include "types/_cuda_char3.h"
#include "types/_cuda_char4.h"

#include "types/_cuda_uchar.h"
#include "types/_cuda_uchar1.h"
#include "types/_cuda_uchar2.h"
#include "types/_cuda_uchar3.h"
#include "types/_cuda_uchar4.h"

#include "types/_cuda_short.h"
#include "types/_cuda_short1.h"
#include "types/_cuda_short2.h"
#include "types/_cuda_short3.h"
#include "types/_cuda_short4.h"

#include "types/_cuda_ushort.h"
#include "types/_cuda_ushort1.h"
#include "types/_cuda_ushort2.h"
#include "types/_cuda_ushort3.h"
#include "types/_cuda_ushort4.h"

#include "types/_cuda_int.h"
#include "types/_cuda_int1.h"
#include "types/_cuda_int2.h"
#include "types/_cuda_int3.h"
#include "types/_cuda_int4.h"

#include "types/_cuda_uint.h"
#include "types/_cuda_uint1.h"
#include "types/_cuda_uint2.h"
#include "types/_cuda_uint3.h"
#include "types/_cuda_uint4.h"

#include "types/_cuda_long.h"
#include "types/_cuda_long1.h"
#include "types/_cuda_long2.h"
#include "types/_cuda_long3.h"
#include "types/_cuda_long4.h"

#include "types/_cuda_ulong.h"
#include "types/_cuda_ulong1.h"
#include "types/_cuda_ulong2.h"
#include "types/_cuda_ulong3.h"
#include "types/_cuda_ulong4.h"

#include "types/_cuda_float.h"
#include "types/_cuda_float1.h"
#include "types/_cuda_float2.h"
#include "types/_cuda_float3.h"
#include "types/_cuda_float4.h"
#include "types/_cuda_float4x4.h"

#ifdef _WIN32
	#include <windows.h>
	#include <stdio.h>
#else
	#include <stdarg.h>
	#include <stdio.h>
	#include <string.h>
#endif

/***********************************************************************************/
/** CONSTANTES                                                                    **/
/***********************************************************************************/

#define PI (3.14159265359f)
#define DEG2RAD (0.01745329252f)
#define RAD2DEG (57.2957795131f)

/***********************************************************************************/
/** MACROS                                                                        **/
/***********************************************************************************/

#define irand( min, max ) rand() % (max - min) + min
#define frand( min, max ) (double)rand() / (double)RAND_MAX * (max - min) + min

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

inline static void _assert( bool predicate, char* file, int line, char* message )
{
	if ( !predicate )
	{
		printf( "Error in %s, line %d : %s\n\n", file, line, message );

		system( "pause" );

		exit( -1 );
	}
}

/***********************************************************************************/

#endif
