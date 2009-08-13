
#ifndef _CUDA_ENGINE
#define _CUDA_ENGINE

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDARaytracer.h"
#include "CUDARenderer.h"
#include "CUDAWindow.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine
	{

/***********************************************************************************/
/** CLASSES AMIES                                                                 **/
/***********************************************************************************/

		friend class CUDARaytracer;
		friend class CUDARenderer;
		friend class CUDAWindow;

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		CUDARaytracer* raytracer;
		CUDARenderer* renderer;
		CUDAWindow* window;

		double frameTime;
		double cycleTime;
		double totalTime;
		double frameDelta;

		wide cycleCount;
		wide totalCount;

		double currentFramerate;
		double averageFramerate;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAEngine()
			Instancie cette classe dont le but est de centraliser l'acc�s aux objets
			communs au module CUDART.
		*/
		CUDAEngine();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		getRaytracer()
			Renvoie le raytracer associ� � cette instance ou NULL si celui-ci n'a
			pas encore �t� cr��.
		*/
		CUDARaytracer* getRaytracer();

		/*
		getRenderer()
			Renvoie le renderer associ� � cette instance ou NULL si celui-ci n'a
			pas encore �t� cr��.
		*/
		CUDARenderer* getRenderer();

		/*
		getWindow()
			Renvoie la fen�tre associ�e � cette instance ou NULL si celle-ci n'a
			pas encore �t� cr��e.
		*/
		CUDAWindow* getWindow();

		/*
		getTime()
			Renvoie le nombre de secondes �coul� depuis le d�marrage du syst�me.
		*/
		double getTime();

		/*
		getDelta()
			Renvoie le temps �coul� pour calculer et afficher la derni�re
			frame (en secondes)
		*/
		double getDelta();

		/*
		getCurrentFramerate()
			Renvoie le nombre de frames affich�es lors de la derni�re seconde.
		*/
		double getCurrentFramerate();

		/*
		getAverageFramerate()
			Renvoie le nombre moyen de frames affich�es par seconde depuis
			l'instanciation de cette classe.
		*/
		double getAverageFramerate();

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		update()
			Met � jour l'ensemble des variables temporelles. Doit �tre appel�e une
			fois par frame.
		*/
		void update();

	};
}

/***********************************************************************************/

#endif
