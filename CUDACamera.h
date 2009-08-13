
#ifndef _CUDA_CAMERA
#define _CUDA_CAMERA

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

/***********************************************************************************/

namespace renderkit
{
	class CUDACamera
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	private:

		float fov;
		float ratio;

		float zNearPlane;
		float zFarPlane;

		float3 dir;

		float3 center;
		float3 angle;

		float4x4 matView;
		float4x4 matProj;
		float4x4 matInvView;
		float4x4 matInvProj;

		bool dirty;

/***********************************************************************************/
/** METHODES PRIVEES                                                              **/
/***********************************************************************************/

	private:

		inline void dirFromAngle();
		inline void angleFromDir();
		
/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDACamera()
			Instancie la classe en définissant une caméra située au point
			{ 0.0f, 0.0f, 0.0f } et regardant en direction de l'axe Z positif. Cette
			caméra est initialisée pour un écran 4/3 et dispose d'un champ de vision
			sur 90°.
		*/
		CUDACamera();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		moveForward( float _speed )
			Déplace cette caméra vers l'avant (défini par le vecteur direction) à
			une vitesse définie par le paramètre _speed.
		*/
		void moveForward( float _speed );

		/*
		moveBackward( float _speed )
			Déplace cette caméra vers l'arrière (défini par le vecteur opposé à
			direction) à une vitesse définie par le paramètre _speed.
		*/
		void moveBackward( float _speed );

		/*
		moveLeft( float _speed )
			Déplace cette caméra vers la gauche (défini par le vecteur perpendiculaire
			à la projection de direction sur le plan XZ) à une vitesse définie par le
			paramètre _speed.
		*/
		void moveLeft( float _speed );

		/*
		moveRight( float _speed )
			Déplace cette caméra vers la droite (défini par le vecteur perpendiculaire
			à la projection de direction sur le plan XZ) à une vitesse définie par le
			paramètre _speed.
		*/
		void moveRight( float _speed );

		/*
		translate( float _x, float _y, float _z )
			Déplace cette caméra d'un certain nombre d'unités sur les axes X, Y, et Z.
		*/
		void translate( float _x, float _y, float _z );

		/*
		rotate( float _x, float _y, float _z )
			Fait pivoter cette caméra d'un certain nombre de radians par rapport aux
			axes X, Y, et Z.
		*/
		void rotate( float _x, float _y, float _z );

		/*
		setCenter( float _x, float _y, float _z )
			Téléporte cette caméra au point { _x, _y, _z } dans le repère global.
		*/
		void setCenter( float _x, float _y, float _z );

		/*
		setCenter( float3 _center )
			Téléporte cette caméra au point { _center.x, _center.y, _center.z } dans
			le repère global.
		*/
		void setCenter( float3 _center );

		/*
		getCenter()
			Renvoie la position de cette caméra dans le repère global.
		*/
		float3 getCenter() { return center; };

		/*
		lookAt( float _x, float _y, float _z )
			Calcule et définit l'angle permettant à cette caméra d'observer l'objet
			situé au point { _x, _y, _z } dans le repère global.
		*/
		void lookAt( float _x, float _y, float _z );

		/*
		lookAt( float3 _point )
			Calcule et définit l'angle permettant à cette caméra d'observer l'objet
			situé au point { _center.x, _center.y, _center.z } dans le repère global.
		*/
		void lookAt( float3 _point );

		/*
		getDir()
			Renvoie le vecteur direction normalisé de cette caméra.
		*/
		float3 getDir() { return dir; };

		/*
		setAngle( float _x, float _y, float _z )
			Définit les angles (en radians) de cette caméra par rapport aux
			axes X (tangage), Y (lacet), et Z (roulis).
		*/
		void setAngle( float _x, float _y, float _z );

		/*
		setAngle( float3 _angle )
			Définit les angles (en radians) de cette caméra par rapport aux
			axes X (tangage), Y (lacet), et Z (roulis).
		*/
		void setAngle( float3 _angle );

		/*
		getAngle()
			Renvoie un vecteur contenant les angles (en radians) de cette caméra
			par rapport aux axes X (tangage), Y (lacet), et Z (roulis).
		*/
		float3 getAngle() { return angle; };

		/*
		setFOV( float _fov )
			Définit l'angle d'ouverture (champ de vision) de cette caméra.
		*/
		void setFOV( float _fov );

		/*
		getFOV()
			Revoie l'angle d'ouverture (champ de vision) de cette caméra.
		*/
		float getFOV() { return fov; };

		/*
		setRatio( float _ratio )
			Définit le rapport largeur/hauteur de l'écran destiné à cette caméra.
		*/
		void setRatio( float _ratio );

		/*
		setRatio( int _width, int _height )
			Définit le rapport largeur/hauteur de l'écran destiné à cette caméra.
		*/
		void setRatio( int _width, int _height );

		/*
		getRatio()
			Renvoie le rapport largeur/hauteur de l'écran destiné à cette caméra.
		*/
		float getRatio() { return ratio; };

		/*
		setZNearPlane( float _zNearPlane )
			Définit la distance autorisée par rapport au plan rapproché (plus cette
			distance est faible plus on augmente les détails situés au premier plan).
		*/
		void setZNearPlane( float _zNearPlane );

		/*
		getZNearPlane()
			Renvoie la distance autorisée par rapport au plan rapproché.
		*/
		float getZNearPlane() { return zNearPlane; };

		/*
		setZFarPlane( float _zFarPlane )
			Définit la distance autorisée par rapport au plan éloigné (plus cette
			distance est grande plus on augmente les détails situés à l'arrière plan).
		*/
		void setZFarPlane( float _zFarPlane );

		/*
		getZFarPlane()
			Renvoie la distance autorisée par rapport au plan éloigné.
		*/
		float getZFarPlane() { return zFarPlane; };

		/*
		pitch()
			Renvoie l'angle (en radians) de cette caméra par rapport à l'axe X (tangage).
		*/
		float pitch() { return angle.x; };

		/*
		yaw()
			Renvoie l'angle (en radians) de cette caméra par rapport à l'axe Y (lacet).
		*/
		float yaw() { return angle.y; };

		/*
		roll()
			Renvoie l'angle (en radians) de cette caméra par rapport à l'axe Z (roulis).
		*/
		float roll() { return angle.z;  };

		/*
		x()
			Renvoie la position (en u) de cette caméra sur l'axe X dans le repère global.
		*/
		float x() { return center.x; };

		/*
		y()
			Renvoie la position (en u) de cette caméra sur l'axe Y dans le repère global.
		*/
		float y() { return center.y; };

		/*
		z()
			Renvoie la position (en u) de cette caméra sur l'axe Z dans le repère global.
		*/
		float z() { return center.z; };

		/*
		dx()
			Renvoie la direction de cette caméra sur l'axe X.
		*/
		float dx() { return dir.x; };

		/*
		dy()
			Renvoie la direction de cette caméra sur l'axe X.
		*/
		float dy() { return dir.y; };

		/*
		dz()
			Renvoie la direction de cette caméra sur l'axe X.
		*/
		float dz() { return dir.z; };

		/*
		getViewMatrix()
			Renvoie la matrice de transformation vers l'espace de la caméra.
		*/
		float4x4 getViewMatrix();

		/*
		getProjMatrix()
			Renvoie la matrice de projection sur l'écran.
		*/
		float4x4 getProjMatrix();

		/*
		getInvViewMatrix()
			Renvoie la matrice inverse de transformation vers l'espace de la caméra.
		*/
		float4x4 getInvViewMatrix();

		/*
		getInvProjMatrix()
			Renvoie la matrice inverse de projection sur l'écran.
		*/
		float4x4 getInvProjMatrix();

/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		update()
			Force la mise à jour des matrices.
		*/
		void update();

	};
}

/***********************************************************************************/

#endif
