
#ifndef _CUDA_MESH
#define _CUDA_MESH

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDAArray.h"
#include "CUDAModel.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDARaytracer;

	class CUDAMesh
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		CUDABox hitbox;

		CUDAArray<float4>* vertices;
		CUDAArray<float4>* normals;
		CUDAArray<float4>* colors;
		CUDAArray<float2>* texcoords;

		CUDAArray<uint4>* faces;

		CUDAModel* model;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAMesh()
			Instancie cette classe dont le but est de centraliser l'accès aux
			ressources relatives à un mesh (scène contenant un ou plusieurs modèles
			3d) en mémoire vidéo.
		*/
		CUDAMesh();

		/*
		~CUDAMesh()
			Libère la mémoire vidéo réservée aux ressources de ce mesh.
		*/
		~CUDAMesh();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getModel()
			Renvoie le modèle 3d associé à cette instance ou NULL si le mesh a été
			directement chargé à l'aide de la méthode quickLoad().
		*/
		CUDAModel* getModel() const { return model; }

		/*
		getVerticesArray()
			Renvoie le tableau contenant les vertices de ce mesh ou bien NULL si ce
			dernier n'en a pas (ce qui est peu probable).
		*/
		CUDAArray<float4>* getVerticesArray() const { return vertices; }

		/*
		getNormalsArray()
			Renvoie le tableau contenant les normales de ce mesh ou bien NULL si ce
			dernier n'en a pas.
		*/
		CUDAArray<float4>* getNormalsArray() const { return normals; }

		/*
		getColorsArray()
			Renvoie le tableau contenant les couleurs de ce mesh ou bien NULL si ce
			dernier n'en a pas.
		*/
		CUDAArray<float4>* getColorsArray() const { return colors; }

		/*
		getTexcoordsArray()
			Renvoie le tableau contenant les coordonnées de texture de ce mesh ou
			bien NULL si ce dernier n'en a pas.
		*/
		CUDAArray<float2>* getTexcoordsArray() const { return texcoords; }

		/*
		getFacesArray()
			Renvoie le tableau contenant les faces de ce mesh ou bien NULL si ce
			dernier n'en a pas (ce qui est peu probable).
		*/
		CUDAArray<uint4>* getFacesArray() const { return faces; }

		/*
		getVertexCount()
			Renvoie le nombre de vertices de ce mesh.
		*/
		int getVertexCount() const { return vertices ? vertices->getUnitCount() : 0; }

		/*
		getFaceCount()
			Renvoie le nombre de faces de ce mesh.
		*/
		int getFaceCount() const { return faces ? faces->getUnitCount() : 0; }

		/*
		getHitbox()
			Renvoie la boite englobante de ce mesh.
		*/
		CUDABox getHitbox() const { return hitbox; }

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		buildFromModel( CUDAModel* _model )
			Alloue la mémoire vidéo nécessaire et construit ce mesh à partir d'un
			modèle 3d.
		*/
		void buildFromModel( CUDAModel* _model );

		/*
		quickLoad( char* path )
			Alloue la mémoire vidéo nécessaire et charge ce mesh à partir d'un
			fichier dans un format spécifique à CUDART.
		*/
		void quickLoad( char* path );

		/*
		quickSave( char* path, bool overwrite = true )
			Sauvegarde ce mesh dans un fichier au format spécifique à CUDART. Un
			fichier dans ce format pouvant contenir plusieurs meshes ainsi que
			plusieurs arbres, il est possible de forcer la réécriture du fichier en
			passant la propriété overwrite à true.
		*/
		void quickSave( char* path, bool overwrite = true );

	};
}

/***********************************************************************************/

#endif
