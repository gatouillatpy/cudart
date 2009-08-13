
#ifndef _CUDA_RAYTRACER
#define _CUDA_RAYTRACER

/***********************************************************************************/
/** INCLUSIONS                                                                    **/
/***********************************************************************************/

#include "CUDACommon.h"

#include <vector>

#include "CUDASurface.h"
#include "CUDACamera.h"
#include "CUDAMesh.h"
#include "CUDAMeshTree.h"
#include "CUDABuffer.h"
#include "CUDAShader.h"

/***********************************************************************************/
/** PROTOTYPES                                                                    **/
/***********************************************************************************/

void CUDA_calcPrimaryRays
(
	renderkit::CUDABuffer* buffer,
	renderkit::CUDACamera* camera
);

void CUDA_raytraceMeshTree
(
	renderkit::CUDABuffer* buffer,
	renderkit::CUDAMesh* mesh,
	renderkit::CUDAMeshTree* tree,
	bool coherency = true
);

void CUDA_rasterizeMeshTree
(
	renderkit::CUDABuffer* buffer,
	renderkit::CUDACamera* camera,
	renderkit::CUDAMesh* mesh,
	renderkit::CUDAMeshTree* tree
);

void CUDA_interpolateData
(
	renderkit::CUDABuffer* buffer,
	renderkit::CUDAMesh* mesh
);

/***********************************************************************************/

namespace renderkit
{
	class CUDAEngine;

	class CUDARaytracer
	{

/***********************************************************************************/
/** ATTRIBUTS                                                                     **/
/***********************************************************************************/

	protected:

		CUDAEngine* engine;

		CUDACamera* internal_camera;
		CUDACamera* current_camera;

		CUDABuffer* internal_buffer;
		CUDABuffer* current_buffer;

		CUDAShader* internal_shader;
		CUDAShader* current_shader;

/***********************************************************************************/
/** CONSTRUCTEURS / DESTRUCTEUR                                                   **/
/***********************************************************************************/

	public:

		/*
		CUDARaytracer( CUDAEngine* _engine )
			Instancie la classe et initialise les ressources internes.
		*/
		CUDARaytracer( CUDAEngine* _engine );

		/*
		~CUDARaytracer()
			Libère les ressources internes.
		*/
		virtual ~CUDARaytracer();

/***********************************************************************************/
/** ACCESSEURS                                                                    **/
/***********************************************************************************/

	public:

		/*
		getEngine()
			Renvoie un pointeur vers le moteur associé à ce raytracer.
		*/
		CUDAEngine* getEngine() { return engine; }

		/*
		getCamera()
			Renvoie un pointeur vers la caméra associée à ce raytracer.
		*/
		CUDACamera* getCamera() { return current_camera; }

		/*
		setCamera( CUDACamera* _camera )
			Associe une caméra à ce raytracer.
		*/
		void setCamera( CUDACamera* _camera ) { current_camera = _camera; }

		/*
		getBuffer()
			Renvoie un pointeur vers le buffer associé à ce raytracer.
		*/
		CUDABuffer* getBuffer() { return current_buffer; }

		/*
		setBuffer( CUDABuffer* _buffer )
			Associe un buffer à ce raytracer.
		*/
		void setBuffer( CUDABuffer* _buffer ) { current_buffer = _buffer; }

		/*
		getShader()
			Renvoie un pointeur vers le shader associé à ce raytracer.
		*/
		CUDAShader* getShader() { return current_shader; }

		/*
		setShader( CUDAShader* _shader )
			Associe un shader à ce raytracer.
		*/
		void setShader( CUDAShader* _shader ) { current_shader = _shader; }

		/***********************************************************************************/
/** METHODES PUBLIQUES                                                            **/
/***********************************************************************************/

	public:

		/*
		initialize()
			Initialisele raytracer et crée un buffer ainsi qu'un shader de base.
		*/
		virtual void initialize();

		/*
		finalize()
			Libère toutes les ressources relatives au raytracer.
		*/
		virtual void finalize();

		/*
		saveRenderSurface( const char* path )
			Sauvegarde le back buffer en fonction du shader et du buffer spécifié.
		*/
		void saveRenderSurface( const char* path );

		/*
		updateRenderSurface()
			Met à jour le back buffer en fonction du shader et du buffer spécifié.
		*/
		void updateRenderSurface();

		/*
		calcPrimaryRays()
			Remplit les surfaces d'entrées (origine et direction des rayons) du
			buffer en fonction des paramètres de camera.
		*/
		void calcPrimaryRays();

		/*
		raytraceMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree )
			Raytrace le mesh en parcourant l'arbre à partir des rayons stockés dans les
			surfaces d'entrées du buffer.
		*/
		void raytraceMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree, bool coherency = true );

		/*
		rasterizeMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree )
			Rasterize les triangles de mesh. Cette méthode, bien qu'étant encore expérimentale,
			peut s'avérer plus efficace que raytraceMeshTree() dans de nombreux cas, tout en
			produisant un résultat similaire. Elle doit toutefois profiter d'une cohérence
			complète des rayons et est donc à réserver au rendu des rayons primaires, ou de
			manière un peu plus approximative au rendu des rayons d'illumination directe en
			prenant pour origine chacune des sources de lumière (principe du shadow mapping).
		*/
		void rasterizeMeshTree( CUDAMesh* mesh, CUDAMeshTree* tree );

		/*
		interpolateOutputSurfaces()
			Calcule des données supplémentaires (points, normales, couleurs, coordonnées de
			texture, indice des matériaux) en fonction des données renvoyées par les noyaux
			de raytracing/rasterization (indice, profondeur, et coordonnées barycentriques
			des faces affichées par pixel).
		*/
		void interpolateOutputSurfaces( CUDAMesh* mesh, bool points = true, bool normals = true,
											bool colors = false, bool texcoords = false, bool materials = false );

	};
}

/***********************************************************************************/

#endif
