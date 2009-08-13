
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
			Instancie cette classe dont le but est de centraliser l'acc�s aux
			ressources relatives � un mesh (sc�ne contenant un ou plusieurs mod�les
			3d) en m�moire vid�o.
		*/
		CUDAMesh();

		/*
		~CUDAMesh()
			Lib�re la m�moire vid�o r�serv�e aux ressources de ce mesh.
		*/
		~CUDAMesh();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getModel()
			Renvoie le mod�le 3d associ� � cette instance ou NULL si le mesh a �t�
			directement charg� � l'aide de la m�thode quickLoad().
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
			Renvoie le tableau contenant les coordonn�es de texture de ce mesh ou
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
			Alloue la m�moire vid�o n�cessaire et construit ce mesh � partir d'un
			mod�le 3d.
		*/
		void buildFromModel( CUDAModel* _model );

		/*
		quickLoad( char* path )
			Alloue la m�moire vid�o n�cessaire et charge ce mesh � partir d'un
			fichier dans un format sp�cifique � CUDART.
		*/
		void quickLoad( char* path );

		/*
		quickSave( char* path, bool overwrite = true )
			Sauvegarde ce mesh dans un fichier au format sp�cifique � CUDART. Un
			fichier dans ce format pouvant contenir plusieurs meshes ainsi que
			plusieurs arbres, il est possible de forcer la r��criture du fichier en
			passant la propri�t� overwrite � true.
		*/
		void quickSave( char* path, bool overwrite = true );

	};
}

/***********************************************************************************/

#endif
