
#ifdef _WIN32

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAW32Window.h"
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

	CUDAW32Window::CUDAW32Window( CUDAEngine* _engine ) : CUDAWindow( _engine )
	{
		ready = false;
	}

	CUDAW32Window::~CUDAW32Window()
	{
		hide();
	}

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	void CUDAW32Window::create( bool _fullscreen )
	{
		fullscreen = _fullscreen;

		BYTE andMask = 0;
		BYTE xorMask = 255;

		wc.cbSize			= sizeof(WNDCLASSEX); 
		wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
		wc.lpfnWndProc		= windowCallback;
		wc.cbClsExtra		= 0L;
		wc.cbWndExtra		= 0L;
		wc.hInstance		= GetModuleHandle( NULL );
		wc.hIcon			= LoadIcon( NULL, IDI_WINLOGO );
		wc.hCursor			= LoadCursor( NULL, IDC_ARROW );
		wc.hbrBackground	= NULL;
		wc.lpszMenuName		= NULL;
		wc.lpszClassName	= "cudart";
		wc.hIconSm			= NULL;
		RegisterClassEx( &wc );

		RECT rect = { 0, 0, width, height };

		if ( fullscreen )
		{
			DEVMODE dmScreenSettings;
			memset( &dmScreenSettings, 0, sizeof(dmScreenSettings) );
			dmScreenSettings.dmSize = sizeof(dmScreenSettings);
			dmScreenSettings.dmPelsWidth = width;
			dmScreenSettings.dmPelsHeight = height;
			dmScreenSettings.dmBitsPerPel = 32;
			dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

			_assert( ChangeDisplaySettings( &dmScreenSettings, CDS_FULLSCREEN ) != DISP_CHANGE_SUCCESSFUL,
				__FILE__, __LINE__, "CUDAW32Window::create() : Unable to apply the requested fullscreen mode." );
		}

		DWORD dwExStyle, dwStyle;

		if ( fullscreen )
		{
			dwExStyle = WS_EX_APPWINDOW;
			dwStyle = WS_POPUP;
			
			ShowCursor( FALSE );
		}
		else
		{
			dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
			dwStyle = WS_OVERLAPPEDWINDOW;
		}

		AdjustWindowRectEx( &rect, dwStyle, FALSE, dwExStyle );

		if ( fullscreen )
		{
			DWORD dwExStyle = WS_EX_APPWINDOW;
			DWORD dwStyle = WS_POPUP;

			AdjustWindowRectEx( &rect, dwStyle, FALSE, dwExStyle );

			ShowCursor( FALSE );
		}
		else
		{
			DWORD dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
			DWORD dwStyle = WS_OVERLAPPEDWINDOW;

			RECT rect = { 0, 0, width, height };

			AdjustWindowRectEx( &rect, dwStyle, FALSE, dwExStyle );

			ShowCursor( TRUE );
		}

		hWnd = CreateWindowEx( dwExStyle, "cudart", "Raytracer CUDA", dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
						left, top, rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, wc.hInstance, NULL );

		_assert( hWnd != NULL, __FILE__, __LINE__, "CUDAW32Window::create() : Unable to create render window." );

		SetWindowLong( hWnd, GWL_USERDATA, (LONG)this );

		ShowWindow( hWnd, SW_SHOWDEFAULT );
		SetForegroundWindow( hWnd );
		SetFocus( hWnd );
		UpdateWindow( hWnd );

		GetClientRect( hWnd, &rect );
		innerWidth = (int)(rect.right - rect.left);
		innerHeight = (int)(rect.bottom - rect.top);

		ready = true;

		active = true;
	}

	void CUDAW32Window::destroy()
	{
		if ( fullscreen )
		{
			ChangeDisplaySettings( NULL, 0 );
			ShowCursor( TRUE );
		}

		DestroyWindow( hWnd );

		UnregisterClass( "cudart", wc.hInstance );
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAW32Window::show()
	{
		if ( visible ) return;

		visible = true;

		synchronizer = CreateSemaphore( NULL, 1, 1, NULL );

		emptyEvent = CreateEvent( NULL, FALSE, FALSE, NULL );

		thread = CreateThread( NULL, 0, (LPTHREAD_START_ROUTINE)windowThread, (LPVOID)this, 0, NULL );

		{
			int k;

			HANDLE emptyEvent = CreateEvent( NULL, FALSE, FALSE, NULL );
			for ( k = 0 ; k < 75 ; k++ )
			{
				WaitForSingleObject( emptyEvent, 40 );

				if ( ready ) break;
			}
			CloseHandle( emptyEvent );

			_assert( k < 75, __FILE__, __LINE__, "CUDAW32Window::show() : Unable to create render window, function time out." );
		}
	}

	void CUDAW32Window::hide()
	{
		if ( !visible ) return;

		visible = false;

		WaitForSingleObject( synchronizer, INFINITE );

		CloseHandle( emptyEvent );

		CloseHandle( synchronizer );

		synchronizer = 0;
	}

/***********************************************************************************/
/** FONCTIONS AMIES                                                               **/
/***********************************************************************************/

	LRESULT WINAPI windowCallback( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
	{
		CUDAW32Window* window = (CUDAW32Window*)GetWindowLong( hWnd, GWL_USERDATA );

		switch( msg )
		{
			case WM_LBUTTONDOWN:
			case WM_RBUTTONDOWN:
			case WM_MBUTTONDOWN:
			case WM_LBUTTONUP:
			case WM_RBUTTONUP:
			case WM_MBUTTONUP:
			case WM_MOUSEMOVE:
			{
				window->mouseX = (int)LOWORD(lParam);
				window->mouseY = (int)HIWORD(lParam);

				if ( GetAsyncKeyState(VK_SHIFT) & 0x8000 )
					window->keyboardContext |= CUDAWindow::KC_SHIFT;
				else
					window->keyboardContext ^= CUDAWindow::KC_SHIFT;

				if ( GetAsyncKeyState(VK_CONTROL) & 0x8000 )
					window->keyboardContext |= CUDAWindow::KC_CTRL;
				else
					window->keyboardContext ^= CUDAWindow::KC_CTRL;

				if ( GetAsyncKeyState(VK_MENU) & 0x8000 )
					window->keyboardContext |= CUDAWindow::KC_ALT;
				else
					window->keyboardContext ^= CUDAWindow::KC_ALT;
			}
		}

		switch( msg )
		{
			case WM_LBUTTONDOWN:
				window->mouseContext |= CUDAWindow::MC_LEFT;
				break;
			case WM_RBUTTONDOWN:
				window->mouseContext |= CUDAWindow::MC_RIGHT;
				break;
			case WM_MBUTTONDOWN:
				window->mouseContext |= CUDAWindow::MC_MIDDLE;
				break;
		}

		switch( msg )
		{
			case WM_KEYUP:
				window->onKeyUp( wParam );
				break;
			case WM_KEYDOWN:
				window->onKeyDown( wParam );
				break;
			case WM_DESTROY:
				window->onClose();
				break;
			case WM_LBUTTONDOWN:
			case WM_RBUTTONDOWN:
			case WM_MBUTTONDOWN:
				window->onMouseDown();
				break;
			case WM_LBUTTONUP:
			case WM_RBUTTONUP:
			case WM_MBUTTONUP:
				window->onMouseUp();
				break;
			case WM_MOUSEMOVE:
				window->onMouseMove();
				break;
		}

		switch( msg )
		{
			case WM_LBUTTONUP:
				window->mouseContext ^= CUDAWindow::MC_LEFT;
				break;
			case WM_RBUTTONUP:
				window->mouseContext ^= CUDAWindow::MC_RIGHT;
				break;
			case WM_MBUTTONUP:
				window->mouseContext ^= CUDAWindow::MC_MIDDLE;
				break;
		}

		return DefWindowProc( hWnd, msg, wParam, lParam );
	}

	DWORD WINAPI windowThread( LPVOID param )
	{
		CUDAW32Window* window = (CUDAW32Window*)param;

		WaitForSingleObject( window->synchronizer, INFINITE );

		window->create();

		{
			MSG msg = {0}; 

			while( window->isVisible() )
			{
				GetMessage( &msg, window->hWnd, 0, 0 );

				TranslateMessage( &msg );
				DispatchMessage( &msg );
			}

			window->destroy();
		}

		ReleaseSemaphore( window->synchronizer, 1, NULL );

		return 0;
	}
}

#endif
