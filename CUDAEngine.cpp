
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAEngine.h"

#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
	#include <unistd.h>
#endif

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

	CUDAEngine::CUDAEngine()
	{
		raytracer = NULL;
		renderer = NULL;
		window = NULL;

		double time = getTime();
		frameTime = time;
		cycleTime = time;
		totalTime = time;
		
		frameDelta = 0.0;

		cycleCount = 0L;
		totalCount = 0L;

		currentFramerate = 0.0;
		averageFramerate = 0.0;
	}

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	CUDAWindow* CUDAEngine::getWindow()
	{
		return window;
	}

	CUDARenderer* CUDAEngine::getRenderer()
	{
		return renderer;
	}

	CUDARaytracer* CUDAEngine::getRaytracer()
	{
		return raytracer;
	}

	double CUDAEngine::getTime()
	{
		#ifdef _WIN32
			wide counter, frequency;
				
			QueryPerformanceCounter( (LARGE_INTEGER*)&counter );
			QueryPerformanceFrequency( (LARGE_INTEGER*)&frequency );
				
			return (double)counter / (double)frequency;
		#else
			static struct timeval tv;
			static struct timezone tz;

			gettimeofday( &tv, &tz );

			return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
		#endif
	}

	double CUDAEngine::getDelta()
	{
		return frameDelta;
	}

	double CUDAEngine::getCurrentFramerate()
	{
		return currentFramerate;
	}

	double CUDAEngine::getAverageFramerate()
	{
		return averageFramerate;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAEngine::update()
	{
		cycleCount++;

		double time = getTime();

		frameDelta = time - frameTime;
		frameTime = time;

		if ( frameTime - cycleTime >= 1.0 )
		{
			totalCount += cycleCount;

			averageFramerate = (double)totalCount / (frameTime - totalTime);
			currentFramerate = (double)cycleCount / (frameTime - cycleTime);

			cycleTime = frameTime;
			cycleCount = 0L;
		}
	}

}
