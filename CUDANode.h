
#ifndef _CUDA_NODE
#define _CUDA_NODE

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

/***********************************************************************************/

namespace renderkit
{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	struct __builtin_align__(16) bvnode
	{

		union 
		{
			int raw_data0;
			int raw_data1;

			struct 
			{
				uint type : 1; // 0 pour node, 1 pour leaf
				uint axis : 2; // 3 si le noeud a été exclu (lors de la phase de clipping)
				uint child_id; // 2^32 nodes 
				// left_child_id = child_id
				// right_child_id = child_id + 1
			} node;

			struct 
			{
				uint type : 1;
				uint obj_count : 31; // max 2^31 triangles referenced by a leaf
				uint obj_id; // 2^32 triangles 
			} leaf;
		}; // 2 int

		float3 box_min;
		float3 box_max;
	    
	}; // 32 bytes

	typedef bvnode CUDANode;

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	/*
	isLeaf( const bvnode& bvn )
		Renvoie true si bvn est une feuille.
	*/
	inline __device__ bool isLeaf( const bvnode& bvn )
	{
		return ( bvn.leaf.type == 1 );
	}

	/*
	isNode( const bvnode& bvn )
		Renvoie true si bvn n'est pas une feuille.
	*/
	inline __device__ bool isNode( const bvnode& bvn )
	{
		return ( bvn.node.type == 0 );
	}

	/*
	getAxis( const bvnode& bvn )
		Renvoie 0, 1, ou 2 en fonction de l'axe de séparation de ce noeud
		(respectivement X, Y, et Z). Attention cette méthode n'est pas valable
		pour les feuilles.
	*/
	inline __device__ int getAxis( const bvnode& bvn )
	{
		// assert( isNode( bvn ) );

		return (int)bvn.node.axis;
	}

	/*
	isDirectionNegative( const bvnode& bvn, const float4& dir )
		Renvoie vrai si la direction passée en paramètre est négative par rapport à
		l'axe de séparation de ce noeud. Attention cette méthode n'est pas valable
		pour les feuilles.
	*/
	inline __device__ bool isDirectionNegative( const bvnode& bvn, const float4& dir )
	{
		// assert( isNode( bvn ) );

		int axis = getAxis( bvn );

		if ( axis == 0 )
			return ( dir.x < 0.f );
		else if ( axis == 1 )
			return ( dir.y < 0.f );
		else
			return ( dir.z < 0.f );
	}

	/*
	getChildren( const bvnode& bvn, const float4& dir, int& near_child_id, int& far_child_id )
		Renvoie l'indice des enfants de ce noeud en déterminant leur ordre de parcours à
		partir de la direction passée en paramètre. Attention cette méthode n'est pas
		valable pour les feuilles.
	*/
	inline __device__ void getChildren( const bvnode& bvn, const float4& dir, int& near_child_id, int& far_child_id )
	{
		// assert( isNode( bvn ) );

		if ( isDirectionNegative( bvn, dir ) )
		{
			near_child_id = bvn.node.child_id + 1;
			far_child_id =  bvn.node.child_id;
		}
		else
		{
			near_child_id = bvn.node.child_id;
			far_child_id = bvn.node.child_id + 1;
		}
	}

	/*
	getObjects( const bvnode& bvn, int& obj_id, int& obj_count )
		Renvoie l'indice du premier ainsi que le nombre d'objets contenus dans cette
		feuille. Attention cette méthode n'est pas valable pour les noeuds.
	*/
	inline __device__ void getObjects( const bvnode& bvn, int& obj_id, int& obj_count )
	{
		// assert( isLeaf( bvn ) );

		obj_id = bvn.leaf.obj_id;
		obj_count = bvn.leaf.obj_count;
	}

}

/***********************************************************************************/

#endif
