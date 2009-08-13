
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
			Instancie cette classe dont le but est de centraliser l'accès aux objets
			communs au module CUDART.
		*/
		CUDAEngine();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

		/*
		getRaytracer()
			Renvoie le raytracer associé à cette instance ou NULL si celui-ci n'a
			pas encore été créé.
		*/
		CUDARaytracer* getRaytracer();

		/*
		getRenderer()
			Renvoie le renderer associé à cette instance ou NULL si celui-ci n'a
			pas encore été créé.
		*/
		CUDARenderer* getRenderer();

		/*
		getWindow()
			Renvoie la fenêtre associée à cette instance ou NULL si celle-ci n'a
			pas encore été créée.
		*/
		CUDAWindow* getWindow();

		/*
		getTime()
			Renvoie le nombre de secondes écoulé depuis le démarrage du système.
		*/
		double getTime();

		/*
		getDelta()
			Renvoie le temps écoulé pour calculer et afficher la dernière
			frame (en secondes)
		*/
		double getDelta();

		/*
		getCurrentFramerate()
			Renvoie le nombre de frames affichées lors de la dernière seconde.
		*/
		double getCurrentFramerate();

		/*
		getAverageFramerate()
			Renvoie le nombre moyen de frames affichées par seconde depuis
			l'instanciation de cette classe.
		*/
		double getAverageFramerate();

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		update()
			Met à jour l'ensemble des variables temporelles. Doit être appelée une
			fois par frame.
		*/
		void update();

	};
}

/***********************************************************************************/

#endif
