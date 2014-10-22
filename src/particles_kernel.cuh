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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    //uint3 gridSize;
    uint numTotalCells;
    uint numTotalVelNodes;
    uint3 numCells;
    uint3 numVelNodes;
    uint null_grid_value;
    uint null_velgrid_value;
    float3 worldOrigin;
	float3 worldSize;
    float3 cellSize;

    uint numBins;
    float binSize;
    uint seededBin;
    float minBin;

    //const static int NUM_DIAM_BINS = 20;
	//#define NUM_DIAM_BINS 75
	#define NUM_DIAM_BINS 50
	//#define NUM_DIAM_BINS 100
	//#define NUM_DIAM_BINS 150
	//#define NUM_DIAM_BINS 70
    float bVols[NUM_DIAM_BINS];
    float bDiams[NUM_DIAM_BINS];

    const static bool SEEDED = false;
    //const static bool SEEDED = true;

	//const static float NUCLEII_PER_M3 = 20000e6;
    //const static float NUCLEII_PER_M3 = 40000e6;
    //const static float NUCLEII_PER_M3 = 60000e6;
    //const static float NUCLEII_PER_M3 = 30000e6;
    //const static float JET_NUCLEII_PER_M3 = 120000e6;
    //const static float JET_NUCLEII_PER_M3 = 0.0;
    const static float NUCLEII_PER_CC = 40000.;
    //const static float NUCLEII_PER_CC = 65000.;
    //const static float JET_NUCLEII_PER_CC = 40000.0;
    const static float JET_NUCLEII_PER_CC = 0.0;
    //const static float JET_NUCLEII_PER_CC = 100000.;
    //const static float JET_NUCLEII_PER_CC = 500000.;

    //const static float JET_NUCLEII_PER_CC = 2.0e6;
    float lambda_h2o;
    float Cp;
    float Cpw;
    float Cw;
    float density_h2o;
    float density_air;
    float conc_nuc_vol;
    float jet_conc_nuc;
    float patm;
    float gamma;
    float Cphi;
    float Mair;
    float Mwater;
    float hwe;

    float h0;
    float hj;
    float ppw0;
    float ppwj;
    float t0;
    float tj;
    float x0;
    float xj;

    uint numBodies;
    uint maxParticlesPerCell;
    // I don't think this got used, but we could use another one, too
    uint minParticlesPerCell;
    uint avgParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;

    // RMK: size of cylindrical slice
    float darc;
    //float celldensity;
    float dt;
    //float density_rho;
    float schmidt;
    float jet_radius;

	//const static int NUM_CELLS_X = 22;
	//const static int NUM_CELLS_X = 29;
	#define NUM_CELLS_X 80 //47	 //24 //29
	//const static int NUM_CELLS_Z = 96;
	//const static int NUM_CELLS_Z = 88;
	#define NUM_CELLS_Z 160 //188 //95 //88
	//for the velocity correction algorithm
	//float dUr[NUM_CELLS_X][NUM_CELLS_Z];
	//float dUz[NUM_CELLS_X][NUM_CELLS_Z];

	//const static int MAX_GRID_SIZE_X = 26;
	const static int MAX_GRID_SIZE_X = 150; //102; //31; //30;
	//const static int MAX_GRID_SIZE_Z = 60;
	const static int MAX_GRID_SIZE_Z = 200; //162; //61; //58;
	//uint3 ncells;
	// these are the coordinates of the cell boundaries
	float Z[MAX_GRID_SIZE_Z];
	float R[MAX_GRID_SIZE_X];
	// these are the coordinates of the (velocity) cell centers
	//float cZ[MAX_GRID_SIZE_Z];
	//float cR[MAX_GRID_SIZE_X];
	//float Ur[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float Uz[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float nut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float eps[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float tke[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float epstke[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float gradRnut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float gradZnut[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	// this will record how often particles need to go in (+) or out (-)
	//float massdt[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float massflux[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float volume[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	// this will record how much time has elapsed since last pushing/popping
	//float parttime[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];

	//for the velocity correction algorithm
	//float dUr[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
	//float dUz[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Z];
};

// cell parameters
/*struct CellParams
{
	const static int MAX_GRID_SIZE = 10;
	uint3 ncells;
	// these are the coordinates of the cell boundaries
	float Z[MAX_GRID_SIZE];
	float R[MAX_GRID_SIZE];
	// these are the coordinates of the cell centers
	float cZ[MAX_GRID_SIZE];
	float cR[MAX_GRID_SIZE];
	float Ur[MAX_GRID_SIZE][MAX_GRID_SIZE];
	float Uz[MAX_GRID_SIZE][MAX_GRID_SIZE];
	float nut[MAX_GRID_SIZE][MAX_GRID_SIZE];
	float eps[MAX_GRID_SIZE][MAX_GRID_SIZE];
	float tke[MAX_GRID_SIZE][MAX_GRID_SIZE];
	// this will record how often particles need to go in (+) or out (-)
	float massdt[MAX_GRID_SIZE][MAX_GRID_SIZE];
	// this will record how much time has elapsed since last pushing/popping
	float parttime[MAX_GRID_SIZE][MAX_GRID_SIZE];
};*/

#endif
