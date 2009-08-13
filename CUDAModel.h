
#ifndef _CUDA_MODEL
#define _CUDA_MODEL

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDABox.h"

#include <vector>

using namespace std;

/***********************************************************************************/

namespace KModel
{
	class Model;
}

namespace renderkit
{

/***********************************************************************************/

	class CUDAModel
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	public:

		bool hasVertices;	// true si ce mod�le contient des sommets
		bool hasNormals;	// true s'il contient des normales associ�es aux sommets
		bool hasColors;		// true s'il contient des couleurs
		bool hasTexcoords;	// true s'il contient des coordonn�es de texture

	private:

		uint vertexCount;
		uint faceCount;

		aabox box;

		float4* vertices;
		float4* normals;
		float4* colors;
		float2* texcoords;

		uint4* faces;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAModel()
			Instancie cette classe dont le but est de centraliser l'acc�s aux
			ressources relatives � un mod�le 3d en m�moire syst�me.
		*/
		CUDAModel();

		/*
		~CUDAModel()
			Lib�re la m�moire syst�me r�serv�e aux ressources de ce mod�le 3d.
		*/
		~CUDAModel();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		setVertexCount( uint _count )
			Sp�cifie le nombre de vertices pour ce mod�le et alloue la m�moire
			syst�me n�cessaire en fonction des propri�t�s hasXXXX.
		*/
		void setVertexCount( uint _count );

		/*
		getVertexCount()
			Renvoie le nombre de vertices de ce mod�le.
		*/
		uint getVertexCount();

		/*
		setFaceCount( uint _count )
			Sp�cifie le nombre de faces pour ce mod�le et alloue la m�moire
			syst�me n�cessaire.
		*/
		void setFaceCount( uint _count );

		/*
		getFaceCount()
			Renvoie le nombre de faces de ce mod�le.
		*/
		uint getFaceCount();

		/*
		getVertexPointer()
			Renvoie un pointeur vers le premier �l�ment du tableau de sommets ou
			NULL si celui-ci n'a pas encore �t� cr��.
		*/
		float4* getVertexPointer();

		/*
		getNormalPointer()
			Renvoie un pointeur vers le premier �l�ment du tableau de normales ou
			bien NULL si celui-ci n'a pas encore �t� cr��.
		*/
		float4* getNormalPointer();

		/*
		getColorPointer()
			Renvoie un pointeur vers le premier �l�ment du tableau de couleurs ou
			NULL si celui-ci n'a pas encore �t� cr��.
		*/
		float4* getColorPointer();

		/*
		getTexcoordPointer()
			Renvoie un pointeur vers le premier �l�ment du tableau de coordonn�es
			de texture ou bien NULL si celui-ci n'a pas encore �t� cr��.
		*/
		float2* getTexcoordPointer();

		/*
		getFacePointer()
			Renvoie un pointeur vers le premier �l�ment du tableau de faces ou
			NULL si celui-ci n'a pas encore �t� cr��.
		*/
		uint4* getFacePointer();

		/*
		setVertex( uint _id, float _x, float _y, float _z )
			D�finit le sommet { _x, _y, _z } ayant pour indice _id dans le tableau.
			La boite englobante du mod�le est mise � jour en cons�quence.
		*/
		void setVertex( uint _id, float _x, float _y, float _z )
		{
			vertices[_id] = make_float4( _x, _y, _z, 0.0f );
		}

		/*
		setVertex( uint _id, float3 _xyz )
			D�finit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du mod�le est mise � jour en cons�quence.
		*/
		void setVertex( uint _id, float3 _xyz )
		{
			vertices[_id] = make_float4( _xyz );
		}

		/*
		setVertex( uint _id, float4 _xyz )
			D�finit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du mod�le est mise � jour en cons�quence.
		*/
		void setVertex( uint _id, float4 _xyz )
		{
			vertices[_id] = _xyz;
		}

		/*
		setVertex( uint _id, const float* _xyz )
			D�finit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du mod�le est mise � jour en cons�quence.
		*/
		void setVertex( uint _id, const float* _xyz )
		{
			vertices[_id] = make_float4( _xyz[0], _xyz[1], _xyz[2], 0.0f );

			box.merge( vertices[_id] );
		}

		/*
		getVertex( uint _id )
			Renvoie le sommet ayant pour indice _id dans le tableau.
		*/
		float3 getVertex( uint _id )
		{
			return make_float3( vertices[_id] );
		}

		/*
		setNormal( uint _id, float _l, float _m, float _n )
			D�finit la normale { _l, _m, _n } associ�e au sommet ayant pour indice _id
			dans le tableau.
		*/
		void setNormal( uint _id, float _l, float _m, float _n )
		{
			normals[_id] = normalize( make_float4( _l, _m, _n, 0.0f ) );
		}

		/*
		setNormal( uint _id, float3 _lmn )
			D�finit la normale { _lmn } ayant pour indice _id dans le tableau.
		*/
		void setNormal( uint _id, float3 _lmn )
		{
			normals[_id] = normalize( make_float4( _lmn ) );
		}

		/*
		setNormal( uint _id, float4 _lmn )
			D�finit la normale { _lmn } ayant pour indice _id dans le tableau.
		*/
		void setNormal( uint _id, float4 _lmn )
		{
			normals[_id] = normalize( _lmn );
		}

		/*
		setNormal( uint _id, const float* _lmn )
			D�finit la normale { _lmn } ayant pour indice _id dans le tableau.
		*/
		void setNormal( uint _id, const float* _lmn )
		{
			normals[_id] = normalize( make_float4( _lmn[0], _lmn[1], _lmn[2], 0.0f ) );
		}

		/*
		getNormal( uint _id )
			Renvoie la normale ayant pour indice _id dans le tableau.
		*/
		float3 getNormal( uint _id )
		{
			return make_float3( normals[_id] );
		}

		/*
		setColor( uint _id, float _r, float _g, float _b, float _a )
			D�finit la couleur { _r, _g, _b, _a } associ�e au sommet ayant pour
			indice _id dans le tableau.
		*/
		void setColor( uint _id, float _r, float _g, float _b, float _a )
		{
			colors[_id] = make_float4( _r, _g, _b, _a );
		}

		/*
		setColor( uint _id, float4 _rgba )
			D�finit la couleur { _rgba } ayant pour indice _id dans le tableau.
		*/
		void setColor( uint _id, float4 _rgba )
		{
			colors[_id] = _rgba;
		}

		/*
		setColor( uint _id, const float* _rgba )
			D�finit la couleur { _rgba } ayant pour indice _id dans le tableau.
		*/
		void setColor( uint _id, const float* _rgba )
		{
			colors[_id] = make_float4( _rgba[0], _rgba[1], _rgba[2], _rgba[3] );
		}

		/*
		getColor( uint _id )
			Renvoie la couleur ayant pour indice _id dans le tableau.
		*/
		float4 getColor( uint _id )
		{
			return colors[_id];
		}

		/*
		setTexcoord( uint _id, float _u, float _v )
			D�finit les coordonn�es de texture { _u, _v } associ�e au sommet ayant
			pour indice _id dans	le tableau.
		*/
		void setTexcoord( uint _id, float _u, float _v )
		{
			texcoords[_id] = make_float2( _u, _v );
		}

		/*
		setTexcoord( uint _id, float2 _uv )
			D�finit les coordonn�es de texture { _uv } ayant pour indice _id
			dans le tableau.
		*/
		void setTexcoord( uint _id, float2 _uv )
		{
			texcoords[_id] = _uv;
		}

		/*
		setTexcoord( uint _id, const float* _uv )
			D�finit les coordonn�es de texture { _uv } ayant pour indice _id
			dans le tableau.
		*/
		void setTexcoord( uint _id, const float* _uv )
		{
			texcoords[_id] = make_float2( _uv[0], _uv[1] );
		}

		/*
		getTexcoord( uint _id )
			Renvoie les coordonn�es de texture ayant pour indice _id dans le tableau.
		*/
		float2 getTexcoord( uint _id )
		{
			return texcoords[_id];
		}

		/*
		setFace( uint _id, uint _a, uint _b, uint _c, uint mat_id )
			D�finit la face ayant pour indice _id dans le tableau. Les param�tres
			_a, _b, et _c repr�sentent les indices des tableaux de sommets, normales,
			couleurs... Le param�tre mat_id repr�sente l'indice d'un mat�riau dans un
			�ventuel futur tableau de mat�riaux.
		*/
		void setFace( uint _id, uint _a, uint _b, uint _c, uint mat_id )
		{
			faces[_id] = make_uint4( _a, _b, _c, mat_id );
		}

		/*
		getFace( uint _id )
			Renvoie la face ayant pour indice _id dans le tableau.
		*/
		uint4 getFace( uint _id )
		{
			return faces[_id];
		}

		/*
		getHitbox()
			Renvoie la boite englobante de ce mod�le 3d.
		*/
		aabox getHitbox()
		{
			return box;
		}

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		insertModel( KModel::Model& model, MAT44 matrix = NULL )
			Ajoute un mod�le 3d de type conforme au module KModel. Le param�tre
			matrix permet de d�finir une transformation lors de l'ajout de ce mod�le.
			Aucune transformation n'est faite si ce param�tre a pour valeur NULL.
			Attention contrairement aux matrices de type float4x4, les matrices de
			type MAT44 sont stock�es selon l'ordre column-major.
		*/
		void insertModel( KModel::Model& model, MAT44 matrix = NULL );

		/*
		buildFromModel( KModel::Model& model, MAT44 matrix = NULL )
			Construit ce mod�le � partir d'un mod�le 3d de type conforme au module
			KModel. Le param�tre matrix permet de d�finir une transformation lors
			de l'ajout de ce mod�le. Aucune transformation n'est faite si ce param�tre
			a pour valeur NULL.
		*/
		void buildFromModel( KModel::Model& model, MAT44 matrix = NULL );

		/*
		buildFromModels( KModel::Model* models, MAT44* matrices, int count )
			Construit ce mod�le � partir d'un tableau de mod�les 3d de type conforme
			au module KModel. Le param�tre matrices repr�sente un tableau de matrices
			de transformation � appliquer � chacun des mod�les 3d. Le nombre d'�l�ments
			dans chacun des deux tableaux est d�fini par le param�tre count. Aucun des
			deux pointeurs pass�s en param�tre ne peut avoir pour valeur NULL.
		*/
		void buildFromModels( KModel::Model* models, MAT44* matrices, int count );

		/*
		createCube( float width = 10.0f, float height = 10.0f, float depth = 10.0f )
			Construit un mod�le de cube align� sur les axes X, Y, Z du rep�re global
			et dont la largeur, hauteur, et profondeur sont d�finies respectivement
			par les param�tres width, height, et depth.
		*/
		void createCube( float width = 10.0f, float height = 10.0f, float depth = 10.0f );

	};
}

/***********************************************************************************/

#endif
