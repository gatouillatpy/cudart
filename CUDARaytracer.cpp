
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include <cuda_runtime.h>

#include "CUDAD3DRenderer.h"
#include "CUDARaytracer.h"
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

	CUDARaytracer::CUDARaytracer( CUDAEngine* _engine )
	{
		engine = _engine;

		engine->raytracer = this;

		internal_camera = new CUDACamera();
		current_camera = internal_camera;
	}

	CUDARaytracer::~CUDARaytracer()
	{
		delete internal_camera;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDARaytracer::initialize()
	{
		int count = 0;
		int i = 0;

		cudaGetDeviceCount( &count );

		_assert( count != 0, __FILE__, __LINE__, "Unable to find any GPU.\n" );

		for ( i = 0 ; i < count ; i++ )
		{
			cudaDeviceProp prop;
			if( cudaGetDeviceProperties( &prop, i ) == cudaSuccess )
			{
				if( prop.major >= 1 ) break;
			}
		}

		_assert( i != count, __FILE__, __LINE__, "Unable to find any GPU supporting CUDA.\n" );

		CUDARenderer* renderer = engine->getRenderer();

		int width = renderer->getSurface()->getWidth();
		int height = renderer->getSurface()->getHeight();

		internal_buffer = new CUDABuffer( width, height );
		current_buffer = internal_buffer;

		internal_shader = new CUDAShader();
		current_shader = internal_shader;
	}

	void CUDARaytracer::finalize()
	{
		if ( internal_shader )
			delete internal_shader;

		if ( internal_buffer )
			delete internal_buffer;
	}

	void CUDARaytracer::saveRenderSurface( const char* path )
	{
		_assert( typeid( *engine->getRenderer() ) == typeid( CUDAD3DRenderer ), __FILE__, __LINE__,
			"CUDARaytracer::saveRenderSurface() : Function not available, invalid renderer type, CUDAD3DRenderer needed." );

		CUDAD3DRenderer* renderer = (CUDAD3DRenderer*)engine->getRenderer();

		renderer->lockSurface();

		CUDARenderSurface<uint>* output = renderer->getSurface();

		current_shader->run( output, current_buffer, current_camera );

		renderer->saveSurface( path, CUDA_FORMAT_JPG, (CUDASurface<byte>*)output );

		renderer->unlockSurface();
	}

	void CUDARaytracer::updateRenderSurface()
	{
		CUDARenderer* renderer = engine->getRenderer();

		renderer->lockSurface();

		CUDARenderSurface<uint>* output = renderer->getSurface();

		current_shader->run( output, current_buffer, current_camera );

		renderer->unlockSurface();
	}

	void CUDARaytracer::calcPrimaryRays()
	{
		CUDA_calcPrimaryRays( current_buffer, current_camera );
	}

	void CUDARaytracer::raytraceMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree, bool coherency )
	{
		if ( mesh->getVertexCount() == 0 ) return;

		CUDA_raytraceMeshTree( current_buffer, mesh, tree, coherency );
	}

	void CUDARaytracer::rasterizeMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree )
	{
		if ( mesh->getVertexCount() == 0 ) return;

		CUDA_rasterizeMeshTree( current_buffer, current_camera, mesh, tree );
	}

	void CUDARaytracer::interpolateOutputSurfaces( CUDAMesh* mesh, bool points, bool normals, bool colors, bool texcoords, bool materials )
	{
		if ( points && !current_buffer->out_points )
			current_buffer->out_points = new CUDASurface<float4>( current_buffer->width, current_buffer->height );

		if ( normals && !current_buffer->out_normals )
			current_buffer->out_normals = new CUDASurface<float4>( current_buffer->width, current_buffer->height );

		if ( colors && !current_buffer->out_colors )
			current_buffer->out_colors = new CUDASurface<float4>( current_buffer->width, current_buffer->height );

		if ( texcoords && !current_buffer->out_texcoords )
			current_buffer->out_texcoords = new CUDASurface<float2>( current_buffer->width, current_buffer->height );

		if ( materials && !current_buffer->out_materials )
			current_buffer->out_materials = new CUDASurface<uint>( current_buffer->width, current_buffer->height );

		CUDA_interpolateData( current_buffer, mesh );
	}
}
