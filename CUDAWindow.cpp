
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAWindow.h"
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

	CUDAWindow::CUDAWindow( CUDAEngine* _engine )
	{
		engine = _engine;

		engine->window = this;

		setLeft();
		setTop();
		setWidth();
		setHeight();

		visible = false;

		keyboardContext = KC_NONE;
		mouseContext = MC_NONE;
	}

	CUDAWindow::~CUDAWindow()
	{
		keyUpCallbacks.clear();
		keyDownCallbacks.clear();

		mouseMoveCallbacks.clear();
		mouseUpCallbacks.clear();
		mouseDownCallbacks.clear();
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	CUDAEngine* CUDAWindow::getEngine()
	{
		return engine;
	}

	void CUDAWindow::setLeft( int _value )
	{
		left = _value;
	}

	int CUDAWindow::getLeft()
	{
		return left;
	}

	void CUDAWindow::setTop( int _value )
	{
		top = _value;
	}

	int CUDAWindow::getTop()
	{
		return top;
	}

	void CUDAWindow::setWidth( int _value )
	{
		width = _value;
	}

	int CUDAWindow::getWidth()
	{
		return width;
	}

	void CUDAWindow::setHeight( int _value )
	{
		height = _value;
	}

	int CUDAWindow::getHeight()
	{
		return height;
	}

	int CUDAWindow::getInnerWidth()
	{
		return innerWidth;
	}

	int CUDAWindow::getInnerHeight()
	{
		return innerHeight;
	}

	bool CUDAWindow::isVisible()
	{
		return visible;
	}

	bool CUDAWindow::isActive()
	{
		return active;
	}

	void CUDAWindow::registerCloseCallback( LPCLOSECALLBACK _pCallback )
	{
		closeCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterCloseCallback( LPCLOSECALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPCLOSECALLBACK>::iterator it = closeCallbacks.begin() ; it != closeCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) closeCallbacks.erase( it );
		}
	}

	void CUDAWindow::registerKeyUpCallback( LPKEYUPCALLBACK _pCallback )
	{
		keyUpCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterKeyUpCallback( LPKEYUPCALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPKEYUPCALLBACK>::iterator it = keyUpCallbacks.begin() ; it != keyUpCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) keyUpCallbacks.erase( it );
		}
	}

	void CUDAWindow::registerKeyDownCallback( LPKEYDOWNCALLBACK _pCallback )
	{
		keyDownCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterKeyDownCallback( LPKEYDOWNCALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPKEYDOWNCALLBACK>::iterator it = keyDownCallbacks.begin() ; it != keyDownCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) keyDownCallbacks.erase( it );
		}
	}

	void CUDAWindow::registerMouseMoveCallback( LPMOUSEMOVECALLBACK _pCallback )
	{
		mouseMoveCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterMouseMoveCallback( LPMOUSEMOVECALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPMOUSEMOVECALLBACK>::iterator it = mouseMoveCallbacks.begin() ; it != mouseMoveCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) mouseMoveCallbacks.erase( it );
		}
	}

	void CUDAWindow::registerMouseUpCallback( LPMOUSEUPCALLBACK _pCallback )
	{
		mouseUpCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterMouseUpCallback( LPMOUSEUPCALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPMOUSEUPCALLBACK>::iterator it = mouseUpCallbacks.begin() ; it != mouseUpCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) mouseUpCallbacks.erase( it );
		}
	}

	void CUDAWindow::registerMouseDownCallback( LPMOUSEDOWNCALLBACK _pCallback )
	{
		mouseDownCallbacks.push_back( _pCallback );
	}

	void CUDAWindow::unregisterMouseDownCallback( LPMOUSEDOWNCALLBACK _pCallback )
	{
		if ( visible == false ) return;

		for ( vector<LPMOUSEDOWNCALLBACK>::iterator it = mouseDownCallbacks.begin() ; it != mouseDownCallbacks.end() ; ++it )
		{
			if ( *it == _pCallback ) mouseDownCallbacks.erase( it );
		}
	}

/***********************************************************************************/
/** EVENEMENTS                                                                    **/
/***********************************************************************************/

	void CUDAWindow::onClose()
	{
		for ( vector<LPCLOSECALLBACK>::iterator it = closeCallbacks.begin() ; it != closeCallbacks.end() ; ++it )
		{
			LPCLOSECALLBACK Callback = *it;

			Callback();
		}

		active = false;
	}

	void CUDAWindow::onKeyUp( int _keyCode )
	{
		for ( vector<LPKEYUPCALLBACK>::iterator it = keyUpCallbacks.begin() ; it != keyUpCallbacks.end() ; ++it )
		{
			LPKEYUPCALLBACK Callback = *it;

			Callback( keyboardContext, _keyCode );
		}
	}

	void CUDAWindow::onKeyDown( int _keyCode )
	{
		for ( vector<LPKEYDOWNCALLBACK>::iterator it = keyDownCallbacks.begin() ; it != keyDownCallbacks.end() ; ++it )
		{
			LPKEYDOWNCALLBACK Callback = *it;

			Callback( keyboardContext, _keyCode );
		}
	}

	void CUDAWindow::onMouseUp()
	{
		for ( vector<LPMOUSEUPCALLBACK>::iterator it = mouseUpCallbacks.begin() ; it != mouseUpCallbacks.end() ; ++it )
		{
			LPMOUSEUPCALLBACK Callback = *it;

			Callback( keyboardContext, mouseContext, mouseX, mouseY );
		}
	}

	void CUDAWindow::onMouseDown()
	{
		for ( vector<LPMOUSEDOWNCALLBACK>::iterator it = mouseDownCallbacks.begin() ; it != mouseDownCallbacks.end() ; ++it )
		{
			LPMOUSEDOWNCALLBACK Callback = *it;

			Callback( keyboardContext, mouseContext, mouseX, mouseY );
		}
	}

	void CUDAWindow::onMouseMove()
	{
		for ( vector<LPMOUSEMOVECALLBACK>::iterator it = mouseMoveCallbacks.begin() ; it != mouseMoveCallbacks.end() ; ++it )
		{
			LPMOUSEMOVECALLBACK Callback = *it;

			Callback( keyboardContext, mouseContext, mouseX, mouseY );
		}
	}
}
