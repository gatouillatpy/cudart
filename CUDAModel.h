
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

		bool hasVertices;	// true si ce modèle contient des sommets
		bool hasNormals;	// true s'il contient des normales associées aux sommets
		bool hasColors;		// true s'il contient des couleurs
		bool hasTexcoords;	// true s'il contient des coordonnées de texture

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
			Instancie cette classe dont le but est de centraliser l'accès aux
			ressources relatives à un modèle 3d en mémoire système.
		*/
		CUDAModel();

		/*
		~CUDAModel()
			Libère la mémoire système réservée aux ressources de ce modèle 3d.
		*/
		~CUDAModel();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		setVertexCount( uint _count )
			Spécifie le nombre de vertices pour ce modèle et alloue la mémoire
			système nécessaire en fonction des propriétés hasXXXX.
		*/
		void setVertexCount( uint _count );

		/*
		getVertexCount()
			Renvoie le nombre de vertices de ce modèle.
		*/
		uint getVertexCount();

		/*
		setFaceCount( uint _count )
			Spécifie le nombre de faces pour ce modèle et alloue la mémoire
			système nécessaire.
		*/
		void setFaceCount( uint _count );

		/*
		getFaceCount()
			Renvoie le nombre de faces de ce modèle.
		*/
		uint getFaceCount();

		/*
		getVertexPointer()
			Renvoie un pointeur vers le premier élément du tableau de sommets ou
			NULL si celui-ci n'a pas encore été créé.
		*/
		float4* getVertexPointer();

		/*
		getNormalPointer()
			Renvoie un pointeur vers le premier élément du tableau de normales ou
			bien NULL si celui-ci n'a pas encore été créé.
		*/
		float4* getNormalPointer();

		/*
		getColorPointer()
			Renvoie un pointeur vers le premier élément du tableau de couleurs ou
			NULL si celui-ci n'a pas encore été créé.
		*/
		float4* getColorPointer();

		/*
		getTexcoordPointer()
			Renvoie un pointeur vers le premier élément du tableau de coordonnées
			de texture ou bien NULL si celui-ci n'a pas encore été créé.
		*/
		float2* getTexcoordPointer();

		/*
		getFacePointer()
			Renvoie un pointeur vers le premier élément du tableau de faces ou
			NULL si celui-ci n'a pas encore été créé.
		*/
		uint4* getFacePointer();

		/*
		setVertex( uint _id, float _x, float _y, float _z )
			Définit le sommet { _x, _y, _z } ayant pour indice _id dans le tableau.
			La boite englobante du modèle est mise à jour en conséquence.
		*/
		void setVertex( uint _id, float _x, float _y, float _z )
		{
			vertices[_id] = make_float4( _x, _y, _z, 0.0f );
		}

		/*
		setVertex( uint _id, float3 _xyz )
			Définit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du modèle est mise à jour en conséquence.
		*/
		void setVertex( uint _id, float3 _xyz )
		{
			vertices[_id] = make_float4( _xyz );
		}

		/*
		setVertex( uint _id, float4 _xyz )
			Définit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du modèle est mise à jour en conséquence.
		*/
		void setVertex( uint _id, float4 _xyz )
		{
			vertices[_id] = _xyz;
		}

		/*
		setVertex( uint _id, const float* _xyz )
			Définit le sommet { _xyz } ayant pour indice _id dans le tableau. La boite
			englobante du modèle est mise à jour en conséquence.
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
			Définit la normale { _l, _m, _n } associée au sommet ayant pour indice _id
			dans le tableau.
		*/
		void setNormal( uint _id, float _l, float _m, float _n )
		{
			normals[_id] = normalize( make_float4( _l, _m, _n, 0.0f ) );
		}

		/*
		setNormal( uint _id, float3 _lmn )
			Définit la normale { _lmn } ayant pour indice _id dans le tableau.
		*/
		void setNormal( uint _id, float3 _lmn )
		{
			normals[_id] = normalize( make_float4( _lmn ) );
		}

		/*
		setNormal( uint _id, float4 _lmn )
			Définit la normale { _lmn } ayant pour indice _id dans le tableau.
		*/
		void setNormal( uint _id, float4 _lmn )
		{
			normals[_id] = normalize( _lmn );
		}

		/*
		setNormal( uint _id, const float* _lmn )
			Définit la normale { _lmn } ayant pour indice _id dans le tableau.
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
			Définit la couleur { _r, _g, _b, _a } associée au sommet ayant pour
			indice _id dans le tableau.
		*/
		void setColor( uint _id, float _r, float _g, float _b, float _a )
		{
			colors[_id] = make_float4( _r, _g, _b, _a );
		}

		/*
		setColor( uint _id, float4 _rgba )
			Définit la couleur { _rgba } ayant pour indice _id dans le tableau.
		*/
		void setColor( uint _id, float4 _rgba )
		{
			colors[_id] = _rgba;
		}

		/*
		setColor( uint _id, const float* _rgba )
			Définit la couleur { _rgba } ayant pour indice _id dans le tableau.
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
			Définit les coordonnées de texture { _u, _v } associée au sommet ayant
			pour indice _id dans	le tableau.
		*/
		void setTexcoord( uint _id, float _u, float _v )
		{
			texcoords[_id] = make_float2( _u, _v );
		}

		/*
		setTexcoord( uint _id, float2 _uv )
			Définit les coordonnées de texture { _uv } ayant pour indice _id
			dans le tableau.
		*/
		void setTexcoord( uint _id, float2 _uv )
		{
			texcoords[_id] = _uv;
		}

		/*
		setTexcoord( uint _id, const float* _uv )
			Définit les coordonnées de texture { _uv } ayant pour indice _id
			dans le tableau.
		*/
		void setTexcoord( uint _id, const float* _uv )
		{
			texcoords[_id] = make_float2( _uv[0], _uv[1] );
		}

		/*
		getTexcoord( uint _id )
			Renvoie les coordonnées de texture ayant pour indice _id dans le tableau.
		*/
		float2 getTexcoord( uint _id )
		{
			return texcoords[_id];
		}

		/*
		setFace( uint _id, uint _a, uint _b, uint _c, uint mat_id )
			Définit la face ayant pour indice _id dans le tableau. Les paramètres
			_a, _b, et _c représentent les indices des tableaux de sommets, normales,
			couleurs... Le paramètre mat_id représente l'indice d'un matériau dans un
			éventuel futur tableau de matériaux.
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
			Renvoie la boite englobante de ce modèle 3d.
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
			Ajoute un modèle 3d de type conforme au module KModel. Le paramètre
			matrix permet de définir une transformation lors de l'ajout de ce modèle.
			Aucune transformation n'est faite si ce paramètre a pour valeur NULL.
			Attention contrairement aux matrices de type float4x4, les matrices de
			type MAT44 sont stockées selon l'ordre column-major.
		*/
		void insertModel( KModel::Model& model, MAT44 matrix = NULL );

		/*
		buildFromModel( KModel::Model& model, MAT44 matrix = NULL )
			Construit ce modèle à partir d'un modèle 3d de type conforme au module
			KModel. Le paramètre matrix permet de définir une transformation lors
			de l'ajout de ce modèle. Aucune transformation n'est faite si ce paramètre
			a pour valeur NULL.
		*/
		void buildFromModel( KModel::Model& model, MAT44 matrix = NULL );

		/*
		buildFromModels( KModel::Model* models, MAT44* matrices, int count )
			Construit ce modèle à partir d'un tableau de modèles 3d de type conforme
			au module KModel. Le paramètre matrices représente un tableau de matrices
			de transformation à appliquer à chacun des modèles 3d. Le nombre d'éléments
			dans chacun des deux tableaux est défini par le paramètre count. Aucun des
			deux pointeurs passés en paramètre ne peut avoir pour valeur NULL.
		*/
		void buildFromModels( KModel::Model* models, MAT44* matrices, int count );

		/*
		createCube( float width = 10.0f, float height = 10.0f, float depth = 10.0f )
			Construit un modèle de cube aligné sur les axes X, Y, Z du repère global
			et dont la largeur, hauteur, et profondeur sont définies respectivement
			par les paramètres width, height, et depth.
		*/
		void createCube( float width = 10.0f, float height = 10.0f, float depth = 10.0f );

	};
}

/***********************************************************************************/

#endif
