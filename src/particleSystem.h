/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
		// RMK: don't deal with gridSize; it's not uniform any more
		//ParticleSystem(uint numParticles, bool bUseOpenGL);
        //ParticleSystem(uint numParticles, uint3 gridSize, float deltatime, bool bUseOpenGL);
		ParticleSystem(uint maxNumParticles, uint numParticles, uint3 numVelNodes, uint3 numCells, float deltatime, bool bUseOpenGL);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_UNIFORM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
            SCALAR,
            BINS
	};

	// step the simulation
	void update(float deltaTime);
	void reset(ParticleConfig config);
	// RMK I might need these here in the ParticleSystem structure
	//int calcNewGridPos(float x, float y, float z);
	int calcVelGridHashHost(int gridPosx, int gridPosy, int gridPosz);
	int calcGridHashHost(int gridPosx, int gridPosy, int gridPosz);
	uint3 calcGridPos(float posx, float posy, float posz);
	uint3 calcVelGridPos(float posx, float posy, float posz);
	//float cellVelVolume(int j);
	float cellVelVolume(uint3 gridPos);
	float cellVolume(uint3 gridPos);
	//float cellVelVolume(uint3 gridPos);
	void boundaryFlux(float* sortedPos, // accept the positions (sorted)
			//uint  *gridParticleHash,
			//uint  *gridParticleIndex,
			uint* cellStart, uint* cellEnd, uint& numParticles, uint numCells);
	void boundaryFluxHybrid(//float* newPos, float* newScalar, float* newDiamBins,
			float* sortedPos, // accept the positions (sorted)
			float* sortedScalar, // accept the positions (sorted)
			float* sortedDiamBins, uint* gridParticleHash,
			uint* gridParticleIndex, uint* cellStart, uint* cellEnd,
			uint maxNumParticles, uint& numParticles,
			float* cellVelDensity);
	//uint   numCells);
	void boundaryFluxHybridVelCells(//float* newPos, float* newScalar, float* newDiamBins,
			float* sortedPos, // accept the positions (sorted)
			float* sortedScalar, // accept the positions (sorted)
			float* sortedDiamBins, uint* gridParticleHash,
			uint* gridParticleIndex, uint* cellStart, uint* cellEnd,
			uint maxNumParticles, uint& numParticles,
			float* cellVelDensity);
	//uint   numCells);
	void balanceCells(
			// float *newPos,
			// float *newScalar,
			float* sortedPos, // accept the positions (sorted)
			float* sortedScalar, float* sortedDiamBins, uint* gridParticleHash,
			uint* gridParticleIndex, uint* cellStart, uint* cellEnd,
			uint maxNumParticles, uint& numParticles); //, uint totalCells);
	//uint   numCells);
    // RMK: mixCurl
    void mixCurl(float *sortedPos,
                 float *sortedScalar,
                 float *sortedDiamBins,
                 //uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 float *densCell);
                 //uint   numParticles,
                 //uint   numCells);
	float* getArray(ParticleArray array);
	void setArray(ParticleArray array, const float* data, int start, int count);
	void saveCells();
	void zeroCells();
	void setAvgCells();

	float interpRho(float,float);
	float interpEpsTKE(float,float);
	float interpUr(float,float);
	float interpUz(float,float);
	float interpGr(float,float);
	float interpGz(float,float);

	int getNumParticles() const {
		return m_numParticles;
	}

	int getMaxNumParticles() const {
		return m_maxNumParticles;
	}

	unsigned int getCurrentReadBuffer() const {
		return m_posVbo;
	}

	unsigned int getColorBuffer() const {
		return m_colorVBO;
	}

	void* getCudaPosVBO() const {
		return (void*) (m_cudaPosVBO);
	}

	void* getCudaColorVBO() const {
		return (void*) (m_cudaColorVBO);
	}

	void dumpGrid();
	void dumpParticles(uint start, uint count);

	void setIterations(int i) {
		m_solverIterations = i;
	}

	void setDamping(float x) {
		m_params.globalDamping = x;
	}

	void setGravity(float x) {
		m_params.gravity = make_float3(0.0f, x, 0.0f);
	}

	void setCollideSpring(float x) {
		m_params.spring = x;
	}

	void setCollideDamping(float x) {
		m_params.damping = x;
	}

	void setCollideShear(float x) {
		m_params.shear = x;
	}

	void setCollideAttraction(float x) {
		m_params.attraction = x;
	}

	void setColliderPos(float3 x) {
		m_params.colliderPos = x;
	}

	float getParticleRadius() {
		return m_params.particleRadius;
	}

	float3 getColliderPos() {
		return m_params.colliderPos;
	}

	float getColliderRadius() {
		return m_params.colliderRadius;
	}

	uint3 getGridSize() {
		return m_params.numCells;
	}

	float3 getWorldOrigin() {
		return m_params.worldOrigin;
	}

	/*float3 getCellSize()
	 {
	 return m_params.cellSize;
	 }*/
	void addSphere(int index, float* pos, float* vel, int r, float spacing);

    float scale1toC(float temperature);
    float scaleCto1(float temperature);
    float scale1toPa(float humidity);
    float scalePato1(float humidity);
    float w2p(float w);
    float ss_calc(float scalar,float *radii,float *bins);
    float ss_calck(float scalar,float *radii,float *bins,float *k,float dt);
    float ss_calc_enthalpy(float scalar,float *radii,float *bins,float particle_mass);
    float ss_calck_enthalpy(float scalar,float *radii,float *bins,float *k,float dt,float particle_mass);
    void drdt_bins(float scalar,float *bins,float dt,float *radii,float particle_mass);
    void RK4(float scalar,float *bins,float dt,float particle_mass);
    void chemistry_serial(float *oldPos,               // output: new velocity
    float *oldScalar,               // input: sorted positions
    float *oldDiamBins,               // input: sorted velocities
    //float *colors,
    int    numParticles);

protected:
	// methods
	ParticleSystem() {
	}

	uint createVBO(uint size);
	void _initialize(int numParticles, uint3& gridSize, float deltaT);
	void _initializeMax(int maxNumParticles, int numParticles,
			uint3& numVelNodes, uint3& numCells, float deltaT);
	void _finalize();
	//void initGrid(uint *size, float spacing, float jitter, uint numParticles);
	void initGrid(uint* size, float spacing, float jitter, uint maxNumParticles,
			uint numParticles);
	// RMK: Add in Cell read
	void readinfile(char filename[], float f[], int n);
	int filecount(char filename[]);

protected:
	// data
	bool m_bInitialized, m_bUseOpenGL;
	uint m_numParticles;
	uint m_maxNumParticles;
	uint m_maxParticlesPerCell;
	uint m_minParticlesPerCell;
	uint m_avgParticlesPerCell;
	float m_minBin;
	float m_binSize;
	uint m_numBins;
	// CPU data
	float* m_hPos; // particle positions
	float* m_hSortedPos;
	float* m_hVel; // particle velocities
	float* m_hScalar; // particle scalars
	float* m_hSortedScalar; // particle scalars
	float* m_hRandom;
	float* m_hDiamBins;
	float* m_hSortedDiamBins;
	float* m_hColors;
	//float *m_hSortedColor;
	uint* m_hParticleHash;
	uint* m_hParticleIndex;
	uint* m_hCellStart;
	uint* m_hVelCellStart;
	uint* m_hCellEnd;
	uint* m_hVelCellEnd;
	// GPU data
	float* m_dPos;
	float* m_dVel;
	float* m_dScalar;
	float* m_dRandom;
	//float *m_dRandom1;
	//float *m_dRandom2;
	float* m_dDiamBins;
	float* m_dColors;

	// Now track the CFD stuff in potentially bigger arrays
    float* m_hUz;
    float* m_hUr;
    float* m_hEpsTKE;
    float* m_hNut;
    float* m_hTKE;
    float* m_hEps;
    float* m_hMassdt;
    float* m_hMassFlux;
    float* m_hGradZNut;
    float* m_hGradRNut;
    float* m_hDUz;
    float* m_hDUr;
    float* m_hRho;
    float* m_dUz;
    float* m_dUr;
    float* m_dEpsTKE;
    float* m_dNut;
    float* m_dGradZNut;
    float* m_dGradRNut;
    float* m_dDUz;
    float* m_dDUr;
    //float* m_dMassdt;
    //float* m_dMassFlux;

	// velocity correction variables
    float* m_hRhoCells;
	float* m_dens;
	float* m_densHist;
	float* m_densCell;
	float* m_densVelCell;
	int m_histLength;
	//float* m_UCorrect;
	int n_iter;
	float m_mass_added;
	float m_mass_removed;

	//float* m_cellVel;
	float* m_cellScalar;
	float* m_cellD10;
	float* m_cellMass;
	uint* m_cellParts;
	float* m_cellRMSScalar;
	float* m_cellAvgScalar;
	float* m_cellDiams;

	//float *m_dSortedColor;
	//float m_bVols[20];
	//float* m_bVols;
	float* m_dSortedPos;
	float* m_dSortedVel;
	float* m_dSortedScalar;
	float* m_dSortedDiamBins;
	// grid data for sorting method
	uint* m_dGridParticleHash; // grid hash value for each particle
	uint* m_dGridParticleIndex; // particle index for each particle
	uint* m_dCellStart; // index of start of each cell in sorted list
	uint* m_dCellEnd; // index of end of cell
	uint* m_dVelCellStart; // index of start of each cell in sorted list
	uint* m_dVelCellEnd; // index of end of cell
	uint m_gridSortBits;
	uint m_posVbo; // vertex buffer object for particle positions
	uint m_colorVBO; // vertex buffer object for colors
	float* m_cudaPosVBO; // these are the CUDA deviceMem Pos
	float* m_cudaColorVBO; // these are the CUDA deviceMem Color
	struct cudaGraphicsResource* m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource* m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange
	// params
	SimParams m_params;
	//RMK add a cell params variable
	//CellParams m_cellparams;
	//uint3 m_gridSize;
	uint3 m_numCells;
	uint3 m_numVelNodes;
	uint m_null_grid_value;
	uint m_null_velgrid_value;
	uint m_numTotalCells;
	uint m_numTotalVelNodes;
	float3 m_worldSize;
	float3 m_worldOrigin;
	// RMK: size of domain in theta direction
	float m_darc;
	//float m_celldensity;
	float m_dt;
	//float m_density_rho;
	float m_schmidt;
	float m_jet_radius;

	int m_loopCounter;
	int m_loopCounterTotal;

	StopWatchInterface* m_timer;
	uint m_solverIterations;
	//static const int MAX_GRID_SIZE_X = 25;
	static const int MAX_GRID_SIZE_X = 150; //31;
	static const int MAX_GRID_SIZE_Z = 200; //61;
	// RMK: add data for cells
	float m_Z[MAX_GRID_SIZE_Z];
	float m_R[MAX_GRID_SIZE_X];
	//uint3 m_ncells;

	//float m_cZ[MAX_GRID_SIZE_Z];
	//float m_cR[MAX_GRID_SIZE_X];
	//float m_Ur[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_Uz[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_nut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_eps[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_tke[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_epstke[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_gradZnut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_gradRnut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	// this will record how often particles need to go in (+) or out (-)
	//float m_massdt[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_massflux[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float m_volume[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	// this will record how much time has elapsed since last pushing/popping
	//float m_parttime[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//RMKrho
	//float m_rho[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];

};

#endif // __PARTICLESYSTEM_H__

// RMK: let's make a cell class
/*
class CellSystem
{
	public:
		CellSystem(uint3 &gridSize);
		//~CellSystem();
	protected: // methods
	//    CellSystem() {}
		void _initialize(uint3 &gridSize);
		void readinfile(char filename[],float f[],int n);
		int filecount(char filename[]);
	protected: // data
		float *m_Z;
		float *m_R;
};
*/
