
/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDAToonShader.h"

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void runCUDAToonShader
(
	renderkit::CUDARenderSurface<uint>* output,
	renderkit::CUDABuffer* input,
	renderkit::CUDACamera* camera,
	renderkit::CUDAToonShader* shader
);

/***********************************************************************************/
/** DEBUG                                                                         **/
/***********************************************************************************/

#include "../CUDADebug.h"

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

	CUDAToonShader::CUDAToonShader() : CUDAShader()
	{
	}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	void CUDAToonShader::run( CUDARenderSurface<uint>* output, CUDABuffer* input, CUDACamera* camera )
	{
		runCUDAToonShader( output, input, camera, this );
	}

}
