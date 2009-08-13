
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDADebug.h"

/***********************************************************************************/
/** OPERATEURS                                                                    **/
/***********************************************************************************/
#ifdef _WIN32
	#ifdef _DEBUG

		void* operator new( size_t nSize, const char* lpszFileName, int nLine )
		{
			return ::operator new( nSize, 1, lpszFileName, nLine );
		}

		void operator delete( void* p, const char* lpszFileName, int nLine )
		{
			return ::operator delete( p, 1, lpszFileName, nLine );
		}

	#endif // _DEBUG
#endif // _WIN32

/***********************************************************************************/
/** CONSTANTES                                                                    **/
/***********************************************************************************/

#define DEBUG_OUTPUT_LENGTH 256

/***********************************************************************************/
/** FONCTIONS                                                                     **/
/***********************************************************************************/

void debugPrint ( char* _format, ... )
{
#ifdef _DEBUG

	va_list list;
	va_start( list, _format );

	static char buffer[DEBUG_OUTPUT_LENGTH] = {0};

#ifdef _WIN32
	#ifdef __STDC_WANT_SECURE_LIB__
		vsprintf_s( buffer, DEBUG_OUTPUT_LENGTH, _format, list );
	#else //__STDC_WANT_SECURE_LIB__
		StringCbVPrintf( buffer, DEBUG_OUTPUT_LENGTH, _format, list );
	#endif //__STDC_WANT_SECURE_LIB__

	OutputDebugString( buffer );
#else
	vfprintf( stderr, _format, list );
#endif // _WIN32

	va_end( list );

#endif //_DEBUG
}

void debugMemoryLeaks ( )
{
#ifdef _WIN32
	_CrtDumpMemoryLeaks();
#endif // _WIN32
}
