
#ifndef _CUDA_BUFFER
#define _CUDA_BUFFER

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDASurface.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDABuffer
	{
		friend class CUDARaytracer;

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		int width;
		int height;

		CUDASurface<uint>* out_faces_id;
		CUDASurface<float2>* out_coords;
		CUDASurface<float>* out_depths;

		CUDASurface<float4>* out_points;
		CUDASurface<float4>* out_normals;
		CUDASurface<float4>* out_colors;
		CUDASurface<float2>* out_texcoords;
		CUDASurface<uint>* out_materials;

		CUDASurface<float4>* in_origins;
		CUDASurface<float4>* in_directions;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDABuffer( int _width, int _height )
			Instancie la classe et alloue la mémoire vidéo pour cinq surfaces de
			taille _width * _height contenant pour chaque pixel : les vecteurs
			d'origine et de direction des rayons, l'indice des faces intersectées,
			la distance du point d'intersection avec la caméra, ainsi que les
			coordonnées barycentriques des triangles intersectés.
		*/
		CUDABuffer( int _width, int _height );

		/*
		~CUDABuffer()
			Libère la mémoire vidéo pour l'ensemble des surfaces instanciées.
		*/
		~CUDABuffer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getWidth()
			Renvoie la largeur des surfaces instanciées (en nombre de pixels).
		*/
		int getWidth() { return width; }

		/*
		getHeight()
			Renvoie la hauteur des surfaces instanciées (en nombre de pixels).
		*/
		int getHeight() { return height; }

		/*
		getInputOriginSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float4 ayant pour composantes en [x,y] :
				(x,y,z)		: origine du rayon intersecteur
				(w)			: sans intérêt
		*/
		CUDASurface<float4>* getInputOriginSurface() const { return in_origins; }

		/*
		getInputDirectionSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float4 ayant pour composantes en [x,y] :
				(x,y,z)		: direction du rayon intersecteur
				(w)			: sans intérêt
		*/
		CUDASurface<float4>* getInputDirectionSurface() const { return in_directions; }

		/*
		getOutputPointSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float4 ayant pour composantes en [x,y] :
				(x,y,z)		: coordonnées du point intersecté
				(w)			: distance du point avec l'origine du rayon
			Remarque : le pointeur renvoyé peut avoir pour valeur NULL
				si cette surface n'a pas préalablement été initialisée
				à l'aide de la méthode interpolateOutputSurfaces de la
				classe CUDARaytracer.
		*/
		CUDASurface<float4>* getOutputPointSurface() const { return out_points; }

		/*
		getOutputNormalSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float4 ayant pour composantes en [x,y] :
				(x,y,z)		: normale interpolée au point intersecté
				(w)			: sans intérêt
			Remarque : le pointeur renvoyé peut avoir pour valeur NULL
				si cette surface n'a pas préalablement été initialisée
				à l'aide de la méthode interpolateOutputSurfaces de la
				classe CUDARaytracer.
		*/
		CUDASurface<float4>* getOutputNormalSurface() const { return out_normals; }

		/*
		getOutputColorSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float4 ayant pour composantes en [x,y] :
				(x,y,z,w)	: valeurs RGBA de la couleur interpolée au point intersecté
			Remarque : le pointeur renvoyé peut avoir pour valeur NULL
				si cette surface n'a pas préalablement été initialisée
				à l'aide de la méthode interpolateOutputSurfaces de la
				classe CUDARaytracer.
		*/
		CUDASurface<float4>* getOutputColorSurface() const { return out_colors; }

		/*
		getOutputTexcoordSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float2 ayant pour composantes en [x,y] :
				(x,y)	: coordonnées de texture UV interpolées au point intersecté
			Remarque : le pointeur renvoyé peut avoir pour valeur NULL
				si cette surface n'a pas préalablement été initialisée
				à l'aide de la méthode interpolateOutputSurfaces de la
				classe CUDARaytracer.
		*/
		CUDASurface<float2>* getOutputTexcoordSurface() const { return out_texcoords; }

		/*
		getOutputMaterialSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type unsigned int ayant pour valeur en [x,y]
			l'indice du matériau de l'objet intersecté.
		Remarque : le pointeur renvoyé peut avoir pour valeur NULL
			si cette surface n'a pas préalablement été initialisée
			à l'aide de la méthode interpolateOutputSurfaces de la
			classe CUDARaytracer.
		*/
		CUDASurface<uint>* getOutputMaterialSurface() const { return out_materials; }

		/*
		getOutputPointSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type unsigned int ayant pour valeur en [x,y]
			l'indice de la face de l'objet intersecté.
		*/
		CUDASurface<uint>* getOutputFaceIdSurface() const { return out_faces_id; }

		/*
		getOutputCoordSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float2 ayant pour composantes en [x,y] :
				(x,y)	: coordonnées barycentriques du triangle intersecté
		*/
		CUDASurface<float2>* getOutputCoordSurface() const { return out_coords; }

		/*
		getOutputDepthSurface()
			Renvoie un pointeur vers l'instance de CUDASurface
			de type float ayant pour valeur en [x,y] la distance 
			du point avec l'origine du rayon.
		*/
		CUDASurface<float>* getOutputDepthSurface() const { return out_depths; }

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		clearInputSurfaces()
			Remplit l'intégralité des surfaces contenant les vecteurs d'origine et
			de direction des rayons avec la valeur { 0.0f, 0.0f, 0.0f, 0.0f }.
		*/
		void clearInputSurfaces();

		/*
		clearOutputSurfaces()
			Remplit l'intégralité des surfaces contenant les valeurs de profondeur,
			les coordonnées barycentriques d'intersection ainsi que l'indice de la
			face intersectée avec les valeurs 0.0f ou 0.
		*/
		void clearOutputSurfaces();

	};
}

/***********************************************************************************/

#endif
