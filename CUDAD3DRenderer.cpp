
#ifdef _WIN32

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAW32Window.h"
#include "CUDAD3DRenderer.h"
#include "CUDARenderer.h"
#include "CUDAEngine.h"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

#include <cuda_d3d9_interop.h>

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

	CUDAD3DRenderer::CUDAD3DRenderer( CUDAEngine* _engine ) : CUDARenderer( _engine )
	{
	}

	CUDAD3DRenderer::~CUDAD3DRenderer()
	{
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	void CUDAD3DRenderer::lockSurface()
	{
#ifndef __DEVICE_EMULATION__
		cudaD3D9MapResources( 1, (LPDIRECT3DRESOURCE9*)&lpRenderTexture );

		cudaD3D9ResourceGetMappedPointer( (void**)&surface->data, lpRenderTexture, 0, 0 );
#endif
	}

	void CUDAD3DRenderer::unlockSurface()
	{
#ifndef __DEVICE_EMULATION__
		cudaD3D9UnmapResources( 1, (LPDIRECT3DRESOURCE9*)&lpRenderTexture );
#endif
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAD3DRenderer::initialize()
	{
		lpD3D = Direct3DCreate9( D3D_SDK_VERSION );

		_assert( lpD3D != NULL, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to initialize Direct3D." );

		_assert( typeid( *engine->getWindow() ) == typeid( CUDAW32Window ), __FILE__, __LINE__,
			"CUDAD3DRenderer::initialize() : Invalid window type, CUDAW32Window needed." );

		CUDAW32Window* window = (CUDAW32Window*)engine->getWindow();

		D3DPRESENT_PARAMETERS d3dpp = {0};
		d3dpp.Windowed						= fullscreen ? false : true;
		d3dpp.SwapEffect					= D3DSWAPEFFECT_COPY;
		d3dpp.BackBufferFormat				= D3DFMT_X8R8G8B8;
		d3dpp.BackBufferWidth				= surface->getWidth();
		d3dpp.BackBufferHeight				= surface->getHeight();
		d3dpp.EnableAutoDepthStencil		= FALSE;
		d3dpp.FullScreen_RefreshRateInHz	= 0;
		d3dpp.PresentationInterval			= verticalSync ? D3DPRESENT_INTERVAL_ONE
													: D3DPRESENT_INTERVAL_IMMEDIATE;

		if ( hardware )
		{
			lpD3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window->getHandle(),
									D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &lpD3DDevice );
		}
		else
		{
			lpD3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_REF, window->getHandle(),
												D3DCREATE_SOFTWARE_VERTEXPROCESSING, &d3dpp, &lpD3DDevice );
		}

		_assert( lpD3DDevice != NULL, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to create Direct3D device." );

		if ( cudaD3D9SetDirect3DDevice( lpD3DDevice ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to link Direct3D device with CUDA." );

		if ( FAILED( lpD3DDevice->CreateTexture( surface->getWidth(), surface->getHeight(), 1, 0,
									D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &lpRenderTexture, NULL ) ) )
			_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to create Direct3D render texture." );

#ifdef __DEVICE_EMULATION__ 
		if ( cudaMalloc( (void**)&surface->data, surface->getWidth() * surface->getHeight() * sizeof(uint) ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to allocate enough video memory." );
#else
		if ( cudaD3D9RegisterResource( lpRenderTexture, cudaD3D9RegisterFlagsNone ) != cudaSuccess )
			_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to map Direct3D render texture with CUDA." );

		cudaD3D9ResourceSetMapFlags( lpRenderTexture, cudaD3D9MapFlagsWriteDiscard );

		cudaD3D9MapResources( 1, (LPDIRECT3DRESOURCE9*)&lpRenderTexture );

		cudaD3D9ResourceGetMappedPointer( (void**)&surface->data, lpRenderTexture, 0, 0 );
		cudaD3D9ResourceGetMappedPitch( (size_t*)&surface->pitch, 0, lpRenderTexture, 0, 0 );
		cudaD3D9ResourceGetMappedSize( (size_t*)&surface->size, lpRenderTexture, 0, 0 );
		cudaMemset( surface->getPointer(), 0xff, surface->getSize() );
		cudaD3D9UnmapResources( 1, (LPDIRECT3DRESOURCE9*)&lpRenderTexture );
#endif
		{
			void* data = NULL;

			float vertices[] =
			{
				-1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
				+1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
				-1.0f, +1.0f, 0.0f, 0.0f, 0.0f,
				+1.0f, +1.0f, 0.0f, 1.0f, 0.0f
			};

			lpD3DDevice->CreateVertexBuffer( sizeof(vertices), D3DUSAGE_WRITEONLY,
				D3DFVF_XYZ | D3DFVF_TEX1, D3DPOOL_DEFAULT, &lpVertexBuffer, NULL );

			lpVertexBuffer->Lock( 0, sizeof(vertices), (void**)&data, 0 );
			CopyMemory( data, vertices, sizeof(vertices) );
			lpVertexBuffer->Unlock();
		}

		{
			void* data = NULL;

			word indices[] = { 0, 1, 2, 3, 2, 1 };

			lpD3DDevice->CreateIndexBuffer( sizeof(indices), D3DUSAGE_WRITEONLY,
				D3DFMT_INDEX16, D3DPOOL_DEFAULT, &lpIndexBuffer, NULL );

			lpIndexBuffer->Lock( 0, sizeof(indices), (void**)&data, 0 );
			CopyMemory( data, indices, sizeof(indices) );
			lpIndexBuffer->Unlock();
		}

		{
			LPD3DXBUFFER code = NULL;

			D3DVERTEXELEMENT9 declaration[] = 
			{
				{ 0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
				{ 0, 12, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
				D3DDECL_END()
			};

			lpD3DDevice->CreateVertexDeclaration( declaration, &lpVertexDeclaration );

			const CHAR vertexShader[] =
				"vs_1_1 \n"\
				\
				"dcl_position v0 \n"\
				"dcl_texcoord v1 \n"\
				\
				"mov oPos, v0 \n"\
				"mov oT0, v1 \n";
			D3DXAssembleShader( vertexShader, strlen(vertexShader), NULL, NULL, NULL, &code, NULL );

			lpD3DDevice->CreateVertexShader( (DWORD*)code->GetBufferPointer(), &lpVertexShader );

			if( code )
				code->Release(); 
		}

		{
			LPD3DXBUFFER code = NULL;

			const CHAR pixelShader[] = 
				"ps_1_1 \n"\
				\
				"tex t0 \n"\
				"mov r0, t0 \n";
			D3DXAssembleShader( pixelShader, strlen(pixelShader), NULL, NULL, NULL, &code, NULL );

			lpD3DDevice->CreatePixelShader( (DWORD*)code->GetBufferPointer(), &lpPixelShader );

			if( code )
				code->Release(); 
		}

		if( FAILED( D3DXCreateFont( lpD3DDevice, 16, 0, FW_BOLD, 0, FALSE,
						DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY,
						DEFAULT_PITCH | FF_DONTCARE, "Arial", &lpFont ) ) )
			_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::initialize() : Unable to create Direct3D font." );
	}

	void CUDAD3DRenderer::finalize()
	{
		if ( lpFont )
			lpFont->Release();

		if ( lpPixelShader )
			lpPixelShader->Release();

		if ( lpVertexShader )
			lpVertexShader->Release();

		if ( lpVertexDeclaration )
			lpVertexDeclaration->Release();

		if ( lpIndexBuffer )
			lpIndexBuffer->Release();

		if ( lpVertexBuffer )
			lpVertexBuffer->Release();

		cudaD3D9UnregisterResource( lpRenderTexture );

		if ( lpRenderTexture )
			lpRenderTexture->Release();

		cudaThreadExit();

		if ( lpD3DDevice )
			lpD3DDevice->Release();

		lpD3DDevice = NULL;

		if ( lpD3D )
			lpD3D->Release();

		lpD3D = NULL;
	}

	void CUDAD3DRenderer::update()
	{
		lpD3DDevice->BeginScene();

		lpD3DDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_CW );
		lpD3DDevice->SetRenderState( D3DRS_FILLMODE, D3DFILL_SOLID );
		lpD3DDevice->SetRenderState( D3DRS_ZFUNC, D3DCMP_ALWAYS );
		lpD3DDevice->SetRenderState( D3DRS_LIGHTING, FALSE );

		lpD3DDevice->SetSamplerState( 0, D3DSAMP_MIPFILTER, D3DTEXF_POINT );
		lpD3DDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_POINT );
		lpD3DDevice->SetSamplerState( 0, D3DSAMP_MAGFILTER, D3DTEXF_POINT );

		lpD3DDevice->SetStreamSource( 0, lpVertexBuffer, 0, 20 );
		lpD3DDevice->SetIndices( lpIndexBuffer );

		lpD3DDevice->SetTexture( 0, lpRenderTexture );

		lpD3DDevice->SetVertexDeclaration( lpVertexDeclaration );

		lpD3DDevice->SetVertexShader( lpVertexShader );
		lpD3DDevice->SetPixelShader( lpPixelShader );

		lpD3DDevice->DrawIndexedPrimitive( D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2 );

		lpD3DDevice->EndScene();

		for ( vector<CUDALabel>::iterator it = labels.begin() ; it != labels.end() ; ++it )
		{
			CUDALabel label = *it;

			RECT rect = { label.x, label.y, 0, 0 };

			lpFont->DrawText( NULL, label.text.c_str(), -1, &rect, DT_NOCLIP,
								D3DXCOLOR( label.r, label.g, label.b, label.a ) );
		}

		lpD3DDevice->Present( NULL, NULL, NULL, NULL );
	}

	void CUDAD3DRenderer::saveSurface( const char* path, const int format, CUDASurface<byte>* surface )
	{
		if ( format == CUDA_FORMAT_RAW )
		{
			CUDARenderer::saveSurface( path, format, surface );
		}
		else
		{
			LPDIRECT3DSURFACE9 pSrcSurface;

			D3DXIMAGE_FILEFORMAT d3dxiff = (D3DXIMAGE_FILEFORMAT)format;

			if ( surface->unit_size == 4 )
				lpD3DDevice->CreateOffscreenPlainSurface( surface->width, surface->height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &pSrcSurface, NULL );
			else if ( surface->unit_size == 8 )
				lpD3DDevice->CreateOffscreenPlainSurface( surface->width, surface->height, D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM, &pSrcSurface, NULL );
			else if ( surface->unit_size == 16 )
				lpD3DDevice->CreateOffscreenPlainSurface( surface->width, surface->height, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM, &pSrcSurface, NULL );
			else
				_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::saveSurface() : Invalid bit depth." );

			D3DLOCKED_RECT d3dlr;

			pSrcSurface->LockRect( &d3dlr, NULL, D3DLOCK_DISCARD );

			_assert( surface->pitch == d3dlr.Pitch, __FILE__, __LINE__, "CUDAD3DRenderer::saveSurface() : Invalid pitch." );

			surface->copyDataToHost( (byte*)d3dlr.pBits );

			pSrcSurface->UnlockRect();

			_assert( D3DXSaveSurfaceToFile( path, d3dxiff, pSrcSurface, NULL, NULL ) == D3D_OK,
				__FILE__, __LINE__, "CUDAD3DRenderer::saveSurface() : Unable to save the specified file." );

			pSrcSurface->Release();
		}
	}

	CUDASurface<byte>* CUDAD3DRenderer::loadSurface( const char* path )
	{
		FILE* file = fopen( path, "rb" );

		_assert( file != NULL, __FILE__, __LINE__, "CUDAD3DRenderer::load() : Invalid file." );

		fseek( file, 0, SEEK_SET );

		int raw_header;

		fread( &raw_header, sizeof(int), 1, file );

		if ( raw_header == 0x194B7EA2 )
		{
			fclose( file );

			return CUDARenderer::loadSurface( path );
		}
		else
		{
			fclose( file );

			CUDASurface<byte>* surface = new CUDASurface<byte>();

			D3DXIMAGE_INFO d3dxii;

			_assert( D3DXGetImageInfoFromFile( path, &d3dxii ) == D3D_OK,
				__FILE__, __LINE__, "CUDAD3DRenderer::loadSurface() : Unable to load the specified file." );

			if ( d3dxii.Format == D3DFMT_X8R8G8B8 )
				surface->unit_size = 4;
			else if ( d3dxii.Format == D3DFMT_A8R8G8B8 )
				surface->unit_size = 4;
			else if ( d3dxii.Format == D3DFMT_A16B16G16R16F )
				surface->unit_size = 8;
			else if ( d3dxii.Format == D3DFMT_A32B32G32R32F )
				surface->unit_size = 16;
			else
				_assert( false, __FILE__, __LINE__, "CUDAD3DRenderer::load() : Invalid pixel format." );

			surface->width = d3dxii.Width;
			surface->height = d3dxii.Height;
			surface->unit_count = surface->width * surface->height;
			surface->size = surface->unit_count * surface->unit_size;

			LPDIRECT3DSURFACE9 pDestSurface = NULL;

			lpD3DDevice->CreateOffscreenPlainSurface( d3dxii.Width, d3dxii.Height, d3dxii.Format, D3DPOOL_SYSTEMMEM, &pDestSurface, NULL );

			D3DXLoadSurfaceFromFile( pDestSurface, NULL, NULL, path, NULL, D3DX_DEFAULT, 0, &d3dxii );

			D3DLOCKED_RECT d3dlr;

			pDestSurface->LockRect( &d3dlr, NULL, D3DLOCK_DISCARD );

			surface->pitch = d3dlr.Pitch;

			surface->initialize();

			surface->copyDataFromHost( (byte*)d3dlr.pBits );

			pDestSurface->UnlockRect();

			pDestSurface->Release();

			return surface;
		}
	}

}

#endif
