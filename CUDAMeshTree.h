
#ifndef _CUDA_MESH_TREE
#define _CUDA_MESH_TREE

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include "CUDAArray.h"
#include "CUDAMesh.h"
#include "CUDABox.h"
#include "CUDANode.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDARaytracer;

	class CUDAMeshTree
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	public:

		CUDAArray<bvnode>* nodes;
		CUDAArray<aabox>* boxes;
		CUDAArray<uint>* indices;

		CUDAMesh* mesh;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDAMeshTree( CUDAMesh* mesh )
			Instancie cette classe dont le but est de centraliser l'acc�s aux
			ressources relatives � un arbre (structure permettant d'acc�l�rer le
			raytracing d'un mesh, utile dans pour CUDART) en m�moire vid�o. Alloue la
			m�moire vid�o n�cessaire aux ressources de cet arbre.
		*/
		CUDAMeshTree( CUDAMesh* mesh );

		/*
		~CUDAMeshTree()
			Lib�re la m�moire vid�o r�serv�e aux ressources de cet arbre.
		*/
		~CUDAMeshTree();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getMesh()
			Renvoie le mesh associ� � cette instance.
		*/
		CUDAMesh* getMesh() const { return mesh; }

		/*
		getHitbox()
			Renvoie la boite englobante du mesh associ� � cet arbre.
		*/
		CUDABox getHitbox() const { return mesh->getHitbox(); }

		/*
		getBoxesArray()
			Renvoie le tableau contenant les boites englobantes de chacune des faces
			du mesh associ� � cet arbre.
		*/
		CUDAArray<aabox>* getBoxesArray() const { return boxes; }

		/*
		getIndicesArray()
			Renvoie le tableau contenant les indices des faces r�ordonn�es du mesh
			associ� � cet arbre.
		*/
		CUDAArray<uint>* getIndicesArray() const { return indices; }

		/*
		getNodesArray()
			Renvoie le tableau contenant les noeuds de l'arbre ou NULL si ce dernier
			n'a pas encore �t� construit.
		*/
		CUDAArray<bvnode>* getNodesArray() const { return nodes; }

		/*
		getNodeCount()
			Renvoie le nombre de noeuds de cet arbre.
		*/
		uint getNodeCount() const { return nodes ? nodes->getUnitCount() : 0; }

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

		/*
		buildMeshTreeLBVH( uint tree_depth = 12 )
			Construit cet � arbre en utilisant la m�thode Lightweight Bounding Volume
			Hierarchy. Cette m�thode consiste � s�parer l'espace en deux de mani�re
			cyclique sur les axes X, Y, Z. Le param�tre tree_depth permet de
			d�terminer le nombre de s�parations � faire (une valeur de 12 permet de
			construire un octree � trois niveaux). Cette m�thode est clairement plus
			rapide (int�ressant pour les sc�nes anim�es) mais l'arbre qui en r�sulte
			peut �tre de plus mauvaise qualit� (jusqu'� 85% de performances sur le 
			parcours de l'arbre par rapport � la m�thode SAH).
		*/
		void buildMeshTreeLBVH( uint tree_depth = 12 );

		/*
		buildMeshTreeSAH()
			Construit cet � arbre en utilisant la m�thode Surface Area Heuristic.
			Cette m�thode construit un arbre de meilleure qualit� en s�parant
			l'espace de mani�re successive suivant l'axe le plus long de la boite
			englobante d'un noeud. La position du plan de s�paration d�pend de la
			surface des primitives r�parties le long de cet axe.
		*/
		void buildMeshTreeSAH();

		/*
		buildMeshTreeHybrid( uint lbvh_tree_depth = 12 )
			Construit cet � arbre en utilisant la m�thode LBVH avec un certain
			nombre de niveaux (d�fini par lbvh_tree_depth) depuis la racine. Puis
			poursuit la construction en utilisant la m�thode SAH. Cette m�thode,
			bien qu'�tant consid�rablement plus lente que LBVH pure s'av�re tout
			de m�me plus rapide que SAH pure et produit un arbre de qualit� quasi
			�quivalente.
		*/
		void buildMeshTreeHybrid( uint lbvh_tree_depth = 12 );

		/*
		updateNodes( bvnode* nodes_buffer, uint node_count )
			Alloue la m�moire vid�o n�cessaire au stockage des noeuds et copie ces
			derniers depuis le buffer du noyau CUDA_buildMeshTreeXXXX.
		*/
		void updateNodes( bvnode* nodes_buffer, uint node_count );

		/*
		quickLoad( char* path )
			Alloue la m�moire vid�o n�cessaire et charge cet arbre � partir d'un
			fichier dans un format sp�cifique � CUDART.
		*/
		void quickLoad( char* path );

		/*
		quickSave( char* path, bool overwrite = true )
			Sauvegarde cet arbre dans un fichier au format sp�cifique � CUDART. Un
			fichier dans ce format pouvant contenir plusieurs meshes ainsi que
			plusieurs arbres, il est possible de forcer la r��criture du fichier en
			passant la propri�t� overwrite � true.
		*/
		void quickSave( char* path, bool overwrite = false );

	};
}

/***********************************************************************************/

#endif
