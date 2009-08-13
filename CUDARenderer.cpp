
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDARenderer.h"
#include "CUDAEngine.h"

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "CUDADebug.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/***********************************************************************************/

namespace renderkit
{

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	CUDARenderer::CUDARenderer( CUDAEngine* _engine )
	{
		engine = _engine;

		engine->renderer = this;

		fullscreen = false;

		verticalSync = false;

		hardware = true;

		surface = new CUDARenderSurface<uint>
		(
			engine->getWindow()->getInnerWidth(),
			engine->getWindow()->getInnerHeight()
		);
	}

	CUDARenderer::~CUDARenderer()
	{
		labels.clear();

		delete surface;
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	CUDAEngine* CUDARenderer::getEngine()
	{
		return engine;
	}

	void CUDARenderer::enableFullscreen( int _width, int _height )
	{
		fullscreen = true;

		delete surface;

		surface = new CUDARenderSurface<uint>
		(
			_width,
			_height
		);
	}

	void CUDARenderer::disableFullscreen()
	{
		fullscreen = false;

		delete surface;

		surface = new CUDARenderSurface<uint>
		(
			engine->getWindow()->getInnerWidth(),
			engine->getWindow()->getInnerHeight()
		);
	}

	void CUDARenderer::enableVSync()
	{
		verticalSync = true;
	}

	void CUDARenderer::disableVSync()
	{
		verticalSync = false;
	}

	void CUDARenderer::useHardware()
	{
		hardware = true;
	}

	void CUDARenderer::useSoftware()
	{
		hardware = false;
	}

	void CUDARenderer::insertLabel( const string& _text, int _x, int _y,
										float _r, float _g, float _b, float _a )
	{
		CUDALabel label = { _text, _x, _y, _r, _g, _b, _a };

		labels.push_back( label );
	}

	void CUDARenderer::clearForeground()
	{
		labels.clear();
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDARenderer::saveSurface( const char* path, const int format, CUDASurface<byte>* surface )
	{
		if ( format == CUDA_FORMAT_RAW )
		{
			FILE* file = fopen( path, "wb" );

			_assert( file != NULL, __FILE__, __LINE__, "CUDASurface::save() : Invalid file." );

			fseek( file, 0, SEEK_SET );

			int raw_header = 0x194B7EA2;

			fwrite( &raw_header, sizeof(int), 1, file );
			fwrite( &surface->unit_count, sizeof(int), 1, file );
			fwrite( &surface->unit_size, sizeof(int), 1, file );
			fwrite( &surface->size, sizeof(int), 1, file );
			fwrite( &surface->width, sizeof(int), 1, file );
			fwrite( &surface->height, sizeof(int), 1, file );
			fwrite( &surface->pitch, sizeof(int), 1, file );

			byte* host_data; cudaMallocHost( (void**)&host_data, surface->size );

			surface->copyDataToHost( host_data );

			fwrite( host_data, sizeof(byte), surface->size, file );

			cudaFreeHost( host_data );

			fclose( file );
		}
		else
		{
			_assert( false, __FILE__, __LINE__,
				"CUDARenderer::save() : Unsupported file format." );
		}
	}

	CUDASurface<byte>* CUDARenderer::loadSurface( const char* path )
	{
		FILE* file = fopen( path, "rb" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDASurface::load() : Invalid file." );

		fseek( file, 0, SEEK_SET );

		int raw_header;

		fread( &raw_header, sizeof(int), 1, file );

		if ( raw_header == 0x194B7EA2 )
		{
			CUDASurface<byte>* surface = new CUDASurface<byte>();

			fread( &surface->unit_count, sizeof(int), 1, file );
			fread( &surface->unit_size, sizeof(int), 1, file );
			fread( &surface->size, sizeof(int), 1, file );
			fread( &surface->width, sizeof(int), 1, file );
			fread( &surface->height, sizeof(int), 1, file );
			fread( &surface->pitch, sizeof(int), 1, file );

			surface->initialize();

			byte* host_data; cudaMallocHost( (void**)&host_data, surface->size );

			fread( host_data, sizeof(byte), surface->size, file );

			surface->copyDataFromHost( host_data );

			cudaFreeHost( host_data );

			fclose( file );

			return surface;
		}
		else
		{
			fclose( file );

			_assert( false, __FILE__, __LINE__,
				"CUDARenderer::load() : Unsupported file format." );

			return NULL;
		}
	}

}
