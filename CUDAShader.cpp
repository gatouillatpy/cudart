
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAShader.h"

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void runCUDAShader
(
	renderkit::CUDARenderSurface<uint>* output,
	renderkit::CUDABuffer* input,
	renderkit::CUDACamera* camera,
	renderkit::CUDAShader* shader
);

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

	CUDAShader::CUDAShader()
	{
		backColor.x = 0.227451f;
		backColor.y = 0.431373f;
		backColor.z = 0.647059f;
		backColor.w = 1.000000f;
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAShader::run( CUDARenderSurface<uint>* output, CUDABuffer* input, CUDACamera* camera )
	{
		runCUDAShader( output, input, camera, this );
	}

}
