
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
			Instancie cette classe dont le but est de centraliser l'accès aux
			ressources relatives à un arbre (structure permettant d'accélérer le
			raytracing d'un mesh, utile dans pour CUDART) en mémoire vidéo. Alloue la
			mémoire vidéo nécessaire aux ressources de cet arbre.
		*/
		CUDAMeshTree( CUDAMesh* mesh );

		/*
		~CUDAMeshTree()
			Libère la mémoire vidéo réservée aux ressources de cet arbre.
		*/
		~CUDAMeshTree();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getMesh()
			Renvoie le mesh associé à cette instance.
		*/
		CUDAMesh* getMesh() const { return mesh; }

		/*
		getHitbox()
			Renvoie la boite englobante du mesh associé à cet arbre.
		*/
		CUDABox getHitbox() const { return mesh->getHitbox(); }

		/*
		getBoxesArray()
			Renvoie le tableau contenant les boites englobantes de chacune des faces
			du mesh associé à cet arbre.
		*/
		CUDAArray<aabox>* getBoxesArray() const { return boxes; }

		/*
		getIndicesArray()
			Renvoie le tableau contenant les indices des faces réordonnées du mesh
			associé à cet arbre.
		*/
		CUDAArray<uint>* getIndicesArray() const { return indices; }

		/*
		getNodesArray()
			Renvoie le tableau contenant les noeuds de l'arbre ou NULL si ce dernier
			n'a pas encore été construit.
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
			Construit cet à arbre en utilisant la méthode Lightweight Bounding Volume
			Hierarchy. Cette méthode consiste à séparer l'espace en deux de manière
			cyclique sur les axes X, Y, Z. Le paramètre tree_depth permet de
			déterminer le nombre de séparations à faire (une valeur de 12 permet de
			construire un octree à trois niveaux). Cette méthode est clairement plus
			rapide (intéressant pour les scènes animées) mais l'arbre qui en résulte
			peut être de plus mauvaise qualité (jusqu'à 85% de performances sur le 
			parcours de l'arbre par rapport à la méthode SAH).
		*/
		void buildMeshTreeLBVH( uint tree_depth = 12 );

		/*
		buildMeshTreeSAH()
			Construit cet à arbre en utilisant la méthode Surface Area Heuristic.
			Cette méthode construit un arbre de meilleure qualité en séparant
			l'espace de manière successive suivant l'axe le plus long de la boite
			englobante d'un noeud. La position du plan de séparation dépend de la
			surface des primitives réparties le long de cet axe.
		*/
		void buildMeshTreeSAH();

		/*
		buildMeshTreeHybrid( uint lbvh_tree_depth = 12 )
			Construit cet à arbre en utilisant la méthode LBVH avec un certain
			nombre de niveaux (défini par lbvh_tree_depth) depuis la racine. Puis
			poursuit la construction en utilisant la méthode SAH. Cette méthode,
			bien qu'étant considérablement plus lente que LBVH pure s'avère tout
			de même plus rapide que SAH pure et produit un arbre de qualité quasi
			équivalente.
		*/
		void buildMeshTreeHybrid( uint lbvh_tree_depth = 12 );

		/*
		updateNodes( bvnode* nodes_buffer, uint node_count )
			Alloue la mémoire vidéo nécessaire au stockage des noeuds et copie ces
			derniers depuis le buffer du noyau CUDA_buildMeshTreeXXXX.
		*/
		void updateNodes( bvnode* nodes_buffer, uint node_count );

		/*
		quickLoad( char* path )
			Alloue la mémoire vidéo nécessaire et charge cet arbre à partir d'un
			fichier dans un format spécifique à CUDART.
		*/
		void quickLoad( char* path );

		/*
		quickSave( char* path, bool overwrite = true )
			Sauvegarde cet arbre dans un fichier au format spécifique à CUDART. Un
			fichier dans ce format pouvant contenir plusieurs meshes ainsi que
			plusieurs arbres, il est possible de forcer la réécriture du fichier en
			passant la propriété overwrite à true.
		*/
		void quickSave( char* path, bool overwrite = false );

	};
}

/***********************************************************************************/

#endif
