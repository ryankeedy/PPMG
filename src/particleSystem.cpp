/*
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

//#include "render_particles.h"  //RMK

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

//#include <boost/array.hpp>
//#include <boost/numeric/odeint.hpp>
//using namespace boost::numeric::odeint;
//typedef boost::array< double , 20 > bin_type;



// RMK need to handle filenames for reading in CFD data
#include <string>
using namespace std;    // Or using std::string;

//#include <armadillo>
//using namespace arma;

/*#include <cstdio>

//extern "C" {
    // LU decomoposition of a general matrix
//    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
//    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
//}

void inverse(double* A, int N)
{
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete IPIV;
    delete WORK;
}*/

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

//ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

inline float AE(float temperature) {
	//const float AE_A = 8.05573;
	//const float AE_B = 1723.64;
	//const float AE_C = 233.076;

	//printf("What up\n");

	//return powf(10.0,(AE_A-AE_B/(temperature+AE_C))) * 133.32239;
	return powf(10.0,(8.05573-1723.64/(temperature+233.076))) * 133.32239;
}

uint3 ParticleSystem::calcVelGridPos(float posx,float posy,float posz)
{
	uint3 gridPos;

    gridPos.x = 0;
    gridPos.y = 0;
    gridPos.z = 0;
    for (int i=0;i<m_params.numVelNodes.x-1;i++) {
    	if (m_R[i+1]>=posx) {
    		gridPos.x = i;
    		break;
    	}
    }
    for (int i=0;i<m_params.numVelNodes.z-1;i++) {
    	if (m_Z[i+1]>=posz) {
    		gridPos.z = i;
    		break;
    	}
    }

    return gridPos;
}

uint3 ParticleSystem::calcGridPos(float posx,float posy,float posz)
{
	uint3 gridPos;
	gridPos.x = floor((posx - m_params.worldOrigin.x) / m_params.cellSize.x);
	gridPos.y = 0;
	gridPos.z = floor((posz - m_params.worldOrigin.z) / m_params.cellSize.z);
	return gridPos;
}

//float ParticleSystem::cellVolume(int j)
float ParticleSystem::cellVolume(uint3 gridPos)
{
	// This assumes x origin is at zero (cylindrical coordinates)
	return m_params.cellSize.z * m_darc/2.0 * (powf(m_params.cellSize.x*(gridPos.x+1),2.0)-powf(m_params.cellSize.x*gridPos.x,2.0));
}

float ParticleSystem::cellVelVolume(uint3 gridPos)
{
	int i = gridPos.x;
	int k = gridPos.z;
	//printf("cell vol: %f %f %f %15.12f\n",m_Z[k],m_R[i],m_darc,(m_Z[k+1]-m_Z[k]) * m_darc/2.0 * (powf(m_R[i+1],2.0)-powf(m_R[i],2.0)));
	return (m_Z[k+1]-m_Z[k]) * m_darc/2.0 * (powf(m_R[i+1],2.0)-powf(m_R[i],2.0));
	// This assumes x origin is at zero (cylindrical coordinates)
	//return m_params.cellSize.z * m_darc/2.0 * (powf(m_params.cellSize.x*(gridPos.x+1),2.0)-powf(m_params.cellSize.x*gridPos.x,2.0));
}

float
ParticleSystem::interpRho(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;

	return (w1*m_hRho[i2*m_numVelNodes.z+k2] +	w2*m_hRho[i1*m_numVelNodes.z+k2] +	w3*m_hRho[i1*m_numVelNodes.z+k1] +	w4*m_hRho[i2*m_numVelNodes.z+k1]) / denom;
}

float
ParticleSystem::interpEpsTKE(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;

	return (w1*m_hEpsTKE[i2*m_numVelNodes.z+k2] +	w2*m_hEpsTKE[i1*m_numVelNodes.z+k2] +	w3*m_hEpsTKE[i1*m_numVelNodes.z+k1] +	w4*m_hEpsTKE[i2*m_numVelNodes.z+k1]) / denom;
}

float
ParticleSystem::interpUz(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;

//    if (z==0.0 && r==0.5*m_params.cellSize.x) {
//    if (z==0.0) {
//    	printf("r,z: %f,%f %d,%d %f %f %f %f %f\n",r,z,i1,k1,m_hUz[i2*m_numVelCells.z+k2],m_hUz[i1*m_numVelCells.z+k2],m_hUz[i1*m_numVelCells.z+k1],m_hUz[i2*m_numVelCells.z+k1],(w1*m_hUz[i2*m_numVelCells.z+k2] +	w2*m_hUz[i1*m_numVelCells.z+k2] +	w3*m_hUz[i1*m_numVelCells.z+k1] +	w4*m_hUz[i2*m_numVelCells.z+k1]) / denom);
//    }

    return (w1*m_hUz[i2*m_numVelNodes.z+k2] +	w2*m_hUz[i1*m_numVelNodes.z+k2] +	w3*m_hUz[i1*m_numVelNodes.z+k1] +	w4*m_hUz[i2*m_numVelNodes.z+k1]) / denom;
}

float
ParticleSystem::interpUr(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;
    //printf("pos: %d %d\n",i1,k1);

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;
	//printf("denoms %f %f %f\n",denom,denomr,denomz);

	return (w1*m_hUr[i2*m_numVelNodes.z+k2] +	w2*m_hUr[i1*m_numVelNodes.z+k2] +	w3*m_hUr[i1*m_numVelNodes.z+k1] +	w4*m_hUr[i2*m_numVelNodes.z+k1]) / denom;
}

float
ParticleSystem::interpGz(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;
    //printf("pos: %d %d\n",i1,k1);

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;
	//printf("denoms %f %f %f\n",denom,denomr,denomz);

	return (w1*m_hGradZNut[i2*m_numVelNodes.z+k2] +	w2*m_hGradZNut[i1*m_numVelNodes.z+k2] +	w3*m_hGradZNut[i1*m_numVelNodes.z+k1] +	w4*m_hGradZNut[i2*m_numVelNodes.z+k1]) / denom;
}

float
ParticleSystem::interpGr(float r,float z) {
    uint3 velGridPos = calcVelGridPos(r,0,z);
    uint i1 = velGridPos.x;
    uint i2 = i1+1;
    uint k1 = velGridPos.z;
    uint k2 = k1+1;
    //printf("pos: %d %d\n",i1,k1);

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = r-m_R[i1];
	w2r = m_R[i2]-r;
	denomr = m_R[i2]-m_R[i1];
	w1z = z-m_Z[k1];
	w2z = m_Z[k2]-z;
	denomz = m_Z[k2]-m_Z[k1];

	float w1, w2, w3, w4;
	float denom;

	w1 = w1r * w1z;
	w2 = w2r * w1z;
	w3 = w2r * w2z;
	w4 = w1r * w2z;
	denom = denomr * denomz;
	//printf("denoms %f %f %f\n",denom,denomr,denomz);

	return (w1*m_hGradRNut[i2*m_numVelNodes.z+k2] +	w2*m_hGradRNut[i1*m_numVelNodes.z+k2] +	w3*m_hGradRNut[i1*m_numVelNodes.z+k1] +	w4*m_hGradRNut[i2*m_numVelNodes.z+k1]) / denom;
}

//ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float deltatime, bool bUseOpenGL) :
ParticleSystem::ParticleSystem(uint maxNumParticles, uint numParticles, uint3 numVelNodes, uint3 numCells, float deltatime, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_hSortedPos(0),
    m_hRandom(0),
    m_hScalar(0),
    m_hSortedScalar(0),
    m_dPos(0),
    m_dVel(0),
    m_dRandom(0),
    //m_dRandom1(0),
    //m_dRandom2(0),
    m_dScalar(0),
    m_dSortedScalar(0),
    m_numCells(numCells),
    m_numVelNodes(numVelNodes),
    m_timer(NULL),
    m_solverIterations(1),
    m_maxNumParticles(maxNumParticles)
{
	printf("In ParticleSystem\n");
	// DomainSize
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big-refine/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big-refine/RVert.txt";
	char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine-lighter/ZVert.txt";
	char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine-lighter/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet62-big-refine-lighter/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet62-big-refine-lighter/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine/RVert.txt";
	//char Zfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet63-big-refine/ZVert.txt";
	//char Rfile[] = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet63-big-refine/RVert.txt";

	m_numTotalCells = m_numCells.x*m_numCells.y*m_numCells.z;
	m_numTotalVelNodes = m_numVelNodes.x*m_numVelNodes.y*m_numVelNodes.z;
    //float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.numVelNodes = m_numVelNodes;
    m_params.numCells = m_numCells;
    m_params.numTotalVelNodes = m_numTotalVelNodes;
    m_params.numTotalCells = m_numTotalCells;
    m_params.numBodies = m_numParticles;

    // DomainSize
    //m_params.particleRadius = 1.0f / 64.0f / 20.0;
    m_params.particleRadius = 1.0f / 64.0f / 40.0;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);
	m_params.worldSize = make_float3(0.11, 1.0f, 0.44);
    m_params.cellSize = make_float3(m_params.worldSize.x / m_numCells.x, m_params.worldSize.y / m_numCells.y, m_params.worldSize.z / m_numCells.z);
    //m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    //float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    //m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    //m_params.celldensity = 1.0;
    m_params.dt = deltatime;

    m_params.darc = 1.0;
    m_params.schmidt = 0.7;
    //m_params.density_rho = 1.0;
    m_params.maxParticlesPerCell = 40;  //60; 40; //10;
    m_params.minParticlesPerCell = 10;  //20; //10; //6;
    //m_params.maxParticlesPerCell = 80; //10;
    //m_params.minParticlesPerCell = 20; //6;
    //m_params.avgParticlesPerCell = (m_params.maxParticlesPerCell+m_params.minParticlesPerCell)/2;
    m_params.avgParticlesPerCell = 20; //40; //20;
    //m_params.avgParticlesPerCell = 40;
    m_params.null_grid_value = m_numTotalCells+1; //m_numCells.x*m_numCells.y*m_numCells.z+1;
    m_params.null_velgrid_value = m_numTotalVelNodes+1; //m_numVelCells.x*m_numVelCells.y*m_numVelCells.z+1;
    // DomainSize
    //m_params.jet_radius = 0.01;
    m_params.jet_radius = 0.0055;

    if (numCells.x != NUM_CELLS_X || numCells.z != NUM_CELLS_Z) {
    	printf("ERROR; CELL MISMATCH!!!\n");
    	cin.ignore();
    }


    printf("That was a bunch of initializing; now bins!\n");

    //m_params.numBins = m_params.NUM_DIAM_BINS;
    m_params.numBins = NUM_DIAM_BINS;
    m_params.minBin =  0.01; //0.1; //0.02;
    m_params.binSize = 0.3; //0.5; //0.1;
    //m_params.binSize = 0.15; //0.5; //0.1;
    //m_params.binSize = 0.2; //0.5; //0.1;
    //m_params.binSize = 0.1; //0.5; //0.1;
    //m_params.seededBin = 6;
    m_params.seededBin = 7;

	for (int n=1;n<m_params.numBins;n++) {
		 m_params.bDiams[n] = n*m_params.binSize;
		 m_params.bVols[n] = 3.14159265359*powf((m_params.bDiams[n]/2.0e6),3.0) *4./3.; // # in cubic meters
	}
	printf("Out of loop\n");
	m_params.bDiams[0] = m_params.minBin;
	m_params.bVols[0] = 3.14159265359*powf((m_params.bDiams[0]/2.0e6),3.0) *4./3.; // # in cubic meters
	printf("All done with bins\n");

	m_params.lambda_h2o = 2257e3;
	m_params.conc_nuc_vol = m_params.NUCLEII_PER_CC;
	m_params.jet_conc_nuc = m_params.JET_NUCLEII_PER_CC;
	m_params.density_h2o = 1000.0;
	m_params.density_air = 1.23;
	m_params.Cp = 1.012e3;  // specific heat capacity of air
	m_params.Cpw = 1.84e3;  // specific heat capacity of water vapor
	m_params.Cw = 4.19e3;  // specific heat capacity of water
	m_params.patm = 101.0e3;
	m_params.gamma = 8.0e-11;
	m_params.Cphi = 3.0; //2.0;
	m_params.Mair = 28.966;  // molar mass of air (g/mol)
	m_params.Mwater = 18.02;  // molar mass of water (g/mol)
	m_params.hwe = 2501.0e3;  // evaporation heat of water

	m_params.t0 = 21.0;
	//m_params.tj = 62.0;
	m_params.tj = 85.0;
	printf("Let's do AE stuff\n");
	// These calcs solve for partial pressures in units of mm Hg
	m_params.ppw0 =   50.0 /100. * powf(10.0,(8.05573-1723.64/(m_params.t0+233.076))); // * 133.32239;
	printf("AE1\n");
	//m_params.hj = 100.0 /100. * AE(m_params.tj);
	m_params.ppwj =   99.0 /100. * powf(10.0,(8.05573-1723.64/(m_params.tj+233.076))); // * 133.32239;
	// X as a function of saturation pressure
	// http://www.engineeringtoolbox.com/humidity-ratio-air-d_686.html
	// 760 mm Hg = 1 atm
	//m_params.x0 = 0.62198 * m_params.ppw0 / (760.0-m_params.ppw0);
	//m_params.xj = 0.62198 * m_params.ppwj / (760.0-m_params.ppwj);
	//m_params.x0 = m_params.Mwater/m_params.Mair * m_params.ppw0 / (760.0-m_params.ppw0);
	//m_params.xj = m_params.Mwater/m_params.Mair * m_params.ppwj / (760.0-m_params.ppwj);

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

	printf("Try reading Z file (%d)\n",m_numVelNodes.z);
	readinfile(Zfile,m_Z,m_numVelNodes.z);
	//for (int i=0;i<m_numVelNodes.z-1;i++) {
	//	m_cZ[i] = (m_Z[i]+m_Z[i+1])/2.0;
	//}

	//float R[gridSize.x+1];
	//float R[MAX_GRID_SIZE];
	printf("Try reading R file (%d)\n",m_numVelNodes.x);
	readinfile(Rfile,m_R,m_numVelNodes.x);
	//for (int i=0;i<m_numVelNodes.x-1;i++) {
	//	m_cR[i] = (m_R[i]+m_R[i+1])/2.0;
	//}
    //m_params.worldSize = make_float3(m_R[m_numVelCells.x]-m_R[0], 1.0f, m_Z[m_numVelCells.z]-m_Z[0]);

	FILE *fp;
	string filename;
	// DomainSize
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big-refine/"; // CFD directory
	string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine-lighter/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet62-big-refine-lighter/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine/"; // CFD directory
	//string directory = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet63-big-refine/"; // CFD directory
	string Uzfile = "UzNodes.txt";
	string Urfile = "UrNodes.txt";
	string nutfile = "nutNodes.txt";
	string epsfile = "epsNodes.txt";
	string tkefile = "kNodes.txt";
	string gradNutZfile = "gradnutZNodes.txt";
	string gradNutRfile = "gradnutRNodes.txt";
	//RMKrho
	string rhofile = "rhoNodes.txt";

	// Arrays to read in CFD data files
/*
	float Uz [m_numVelCells.x*m_numVelCells.z];
	float Ur [m_numVelCells.x*m_numVelCells.z];
	float tke[m_numVelCells.x*m_numVelCells.z];
	float eps[m_numVelCells.x*m_numVelCells.z];
	float nut[m_numVelCells.x*m_numVelCells.z];
	float gradNutZ[m_numVelCells.x*m_numVelCells.z];
	float gradNutR[m_numVelCells.x*m_numVelCells.z];
*/
	m_hUz       = new float[m_numTotalVelNodes];
	m_hUr       = new float[m_numTotalVelNodes];
	m_hEpsTKE   = new float[m_numTotalVelNodes];
	m_hNut      = new float[m_numTotalVelNodes];
	m_hMassdt   = new float[m_numTotalCells];
	m_hMassFlux = new float[m_numTotalCells];
	m_hGradZNut = new float[m_numTotalVelNodes];
	m_hGradRNut = new float[m_numTotalVelNodes];

	m_hDUz      = new float[m_numTotalCells];
	m_hDUr      = new float[m_numTotalCells];
	memset(m_hDUz,0,m_numTotalCells*sizeof(float));
	memset(m_hDUr,0,m_numTotalCells*sizeof(float));
	//for (int i=0;i<m_numTotalCells;i++) {
	//	m_hDUz[i] = 0.0;
	//	m_hDUr[i] = 0.0;
	//}

	m_hTKE      = new float[m_numTotalVelNodes];
	m_hEps      = new float[m_numTotalVelNodes];
	//RMKrho
	//float rho[m_numVelCells.x*m_numVelCells.z];
	m_hRho      = new float[m_numTotalVelNodes];
	m_hRhoCells = new float[m_numTotalCells];

	// Arrays to store statistics of PDF model
	//float m_cellMass  [m_numCells.x][m_numCells.z];
	//float m_cellScalar[m_numCells.x][m_numCells.z];
	//float m_cellD10   [m_numCells.x][m_numCells.z];

	//strcpy(filename,directory);	strcat(filename,Uzfile);
	filename = directory+Uzfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in Uz\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hUz[i]);
	}
	fclose(fp);

	//strcpy(filename,directory);	strcat(filename,Urfile);
	filename = directory+Urfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in Ur\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hUr[i]);
	}
	fclose(fp);

	filename = directory+nutfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in nut\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hNut[i]);
		//printf("nutin %2d, %2d %e %d\n",i/m_numVelCells.z,i%m_numVelCells.z,nut[i],m_numVelCells.z);
	}
	fclose(fp);

	filename = directory+tkefile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in tke\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hTKE[i]);
	}
	fclose(fp);

	filename = directory+epsfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in eps\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hEps[i]);
		m_hEpsTKE[i] = m_hEps[i]/m_hTKE[i];
	}
	fclose(fp);

	filename = directory+gradNutRfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in gradnutr\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hGradRNut[i]);
		//printf("gradnutrin %2d, %2d %e %d\n",i/m_numVelCells.z,i%m_numVelCells.z,gradNutR[i],m_numVelCells.z);
	}
	fclose(fp);

	filename = directory+gradNutZfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in gradnutz\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hGradZNut[i]);
		//printf("gradnutzin %2d, %2d %e %d\n",i/m_numVelCells.z,i%m_numVelCells.z,gradNutZ[i],m_numVelCells.z);
	}
	fclose(fp);

	//RMKrho

	filename = directory+rhofile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in rho file\n");
	for (int i=0;i<m_numVelNodes.x*m_numVelNodes.z;i++) {
		fscanf(fp,"%f",&m_hRho[i]);
	}
	fclose(fp);

/*
	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {
			m_Uz [i][k] = Uz [i*m_numVelCells.z+k];
			m_Ur [i][k] = Ur [i*m_numVelCells.z+k];
			//m_eps[i][k] = eps[i*m_numVelCells.z+k];
			//m_tke[i][k] = tke[i*m_numVelCells.z+k];
			m_epstke[i][k] = eps[i*m_numVelCells.z+k]/tke[i*m_numVelCells.z+k];
			m_nut[i][k] = nut[i*m_numVelCells.z+k];
			m_massdt[i][k] = 0.0;
			m_massflux[i][k] = 0.0;
			m_gradZnut[i][k] = gradNutZ[i*m_numVelCells.z+k];
			m_gradRnut[i][k] = gradNutR[i*m_numVelCells.z+k];
			//RMKrho
			//m_rho[i][k] = rho[i*m_numVelCells.z+k];
		}
	}
*/
	float flow_out = 0.0;
	float flow_in = 0.0;
	float mass_totes = 0.0;
	m_mass_added = 0.0;
	m_mass_removed = 0.0;
	// RMK: do particle in/out time elapse for cells
	// let's do for cells instead of vel cells
	int num_subdiv = 1.;
	printf("test %f\n",interpUr(0.0,0.0));
	//cin.ignore();
	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
/**/
			if (i == m_numCells.x-1) {
				for (int j=0;j<num_subdiv;j++) {
					m_hMassdt[i*m_numCells.z+k] += -interpUr((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z)*((1.0/num_subdiv)*m_params.cellSize.z)*m_params.darc* (i+1)*m_params.cellSize.x * interpRho((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z) * m_params.dt;
					mass_totes                  += -interpUr((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z)*((1.0/num_subdiv)*m_params.cellSize.z)*m_params.darc* (i+1)*m_params.cellSize.x * interpRho((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z) * m_params.dt;
					if (interpUr((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z)>0.0) {
						flow_out -=                -interpUr((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z)*((1.0/num_subdiv)*m_params.cellSize.z)*m_params.darc* (i+1)*m_params.cellSize.x * interpRho((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z) * m_params.dt;
					} else {
						flow_in  +=                -interpUr((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z)*((1.0/num_subdiv)*m_params.cellSize.z)*m_params.darc* (i+1)*m_params.cellSize.x * interpRho((i+1)*m_params.cellSize.x,(k+(j+0.5)/num_subdiv)*m_params.cellSize.z) * m_params.dt;
					}
				}
			}

			// don't need to do i==0 because that's symmetry boundary
			if (k == 0) {
				for (int j=0;j<num_subdiv;j++) {
					m_hMassdt[i*m_numCells.z+k] += interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z) * m_params.dt;
					mass_totes                  += interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z) * m_params.dt;
					if (interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z)>0.0) {
						flow_in  +=                interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z) * m_params.dt;
					} else {
						flow_out -=                interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,k*m_params.cellSize.z) * m_params.dt;
					}
				}
			}

			if (k == m_numCells.z-1) {
				for (int j=0;j<num_subdiv;j++) {
					m_hMassdt[i*m_numCells.z+k] += -interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z) * m_params.dt;
					mass_totes                  += -interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z) * m_params.dt;
					if (interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z)>0.0) {
						flow_out -=                -interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z) * m_params.dt;
					} else {
						flow_in  +=                -interpUz((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z)*m_params.darc/2.0*(powf((i+(j+1)/num_subdiv)*m_params.cellSize.x,2)-powf((i+j/num_subdiv)*m_params.cellSize.x,2)) * interpRho((i+(j+0.5)/num_subdiv)*m_params.cellSize.x,(k+1)*m_params.cellSize.z) * m_params.dt;
					}
				}
			}

		}
	}

	for (int k=0; k<m_numVelNodes.z; k++) {
	    m_params.Z[k] = m_Z[k];
	    //m_params.cZ[k] = m_cZ[k];
	}
    for (int i=0; i<m_numVelNodes.x; i++) {
        m_params.R[i] = m_R[i];
        //m_params.cR[i] = m_cR[i];
        //printf("Defining cR: %f\n",m_cR[i]);
    }

	m_params.x0 = m_params.Mwater*m_params.ppw0 / (m_params.Mair*(760.0-m_params.ppw0) + m_params.Mwater*m_params.ppw0);
	float mass_vapor = m_params.Mwater*m_params.ppwj / (m_params.Mair*(760.0-m_params.ppwj) + m_params.Mwater*m_params.ppwj);
	printf("lets call interprho %e\n",mass_vapor);
	float mass_air = (1.0-mass_vapor)*interpRho(0,0);
	printf("did i make it here\n");
	mass_vapor = mass_vapor*interpRho(0,0);
	float mass_water;
	if (m_params.SEEDED) {
		mass_water = (m_params.bVols[m_params.seededBin]-m_params.bVols[0]) * m_params.density_h2o * m_params.jet_conc_nuc*1.0e6;
		//printf("mass water = %f %e %e\n",mass_water,m_params.jet_conc_nuc*cellVolume(make_uint3(0,0,0))/m_params.avgParticlesPerCell*1.0e6,cellVolume(make_uint3(1,0,1)));
	} else {
		mass_water = 0.0;
	}
	m_params.xj = (mass_water+mass_vapor)/(mass_water+mass_air+mass_vapor);
	//m_params.xj = m_params.Mwater*m_params.ppwj / (m_params.Mair*(760.0-m_params.ppwj) + m_params.Mwater*m_params.ppwj);
	float Xs = mass_water/(mass_water+mass_air+mass_vapor);
	m_params.h0 = (1.0-m_params.x0)*m_params.Cp*m_params.t0 + m_params.x0*(m_params.Cpw*m_params.t0+m_params.hwe);
	m_params.hj = (1.0-m_params.xj)*m_params.Cp*m_params.tj + m_params.xj*(m_params.Cpw*m_params.tj+m_params.hwe)+Xs*m_params.Cw*m_params.tj;
	printf("AE2\n");

	printf("h0,hj %f, %f   x0,xj,xs %f %f %f   pp0,ppw %f %f\n",m_params.h0,m_params.hj,m_params.x0,m_params.xj,Xs,m_params.ppw0,m_params.ppwj);
	printf("Compare in %e to out %e (%e) %f\n",flow_in,flow_out,mass_totes,flow_in/flow_out);
	//cin.ignore();
	// This is to make sure in == out
	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
			if (m_hMassdt[i*m_numCells.z+k]<0.0) {
				m_hMassdt[i*m_numCells.z+k] = m_hMassdt[i*m_numCells.z+k] * flow_in/flow_out;
			}
		}
	}

    printf("Initializing...\n");
    //_initialize(numParticles,gridSize,deltatime);
    _initializeMax(maxNumParticles,numParticles,numVelNodes,numCells,deltatime);
    printf("Done initializing\n");

}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 8;
    float c[ncolors][3] =
    {
        { 0.0, 0.0, 0.0, },
        { 1.0, 0.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 1.0, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.0, 0.0, },
    	{ 1.0, 1.0, 1.0, },
    };
    //t = t*0.5;
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    if (i>=ncolors-1) {
    	r[0] = 0.7; r[1] = 0.7; r[2] = 0.7;
    } else if (i<0) {
    	//r[0] = 0.3; r[1] = 0.3; r[2] = 0.3;
    	r[0] = 0.5; r[1] = 0.2; r[2] = 0.1;
    } else {
		r[0] = lerp(c[i][0], c[i+1][0], u);
		r[1] = lerp(c[i][1], c[i+1][1], u);
		r[2] = lerp(c[i][2], c[i+1][2], u);
    }
}

void
ParticleSystem::readinfile(char filename[],float f[],int n) {
	FILE *fp;
	//int counter=0;
	fp = fopen(filename,"r");
	for (int i=0;i<n;i++) {
		fscanf(fp,"%f\n",&f[i]);
		printf("%f\n",f[i]);
	}
	fclose(fp);
	//return 0;
}


void
ParticleSystem::_initializeMax(int maxNumParticles, int numParticles, uint3 &numVelNodes, uint3 &numCells, float deltaT)
{
    assert(!m_bInitialized);

    printf("In initializeMax\n");
    m_numParticles = numParticles;
    m_maxNumParticles = maxNumParticles;

    // RMK: set the size of the cylindrical domain in the y/theta direction
    m_darc = 1.0;  // this is assumed to be in radians everywhere
    //m_maxParticlesPerCell = 10;
    //m_minParticlesPerCell = 6;
    //m_avgParticlesPerCell = (m_maxParticlesPerCell+m_minParticlesPerCell)/2;
    //m_maxParticlesPerCell = m_params.maxParticlesPerCell;
    //m_minParticlesPerCell = 6;
    //m_avgParticlesPerCell = (m_maxParticlesPerCell+m_minParticlesPerCell)/2;
    //m_celldensity = 1.0;
    m_schmidt = 0.7;
    m_dt = deltaT;
    //m_density_rho = 1.0;
    m_null_grid_value    = numCells.x*numCells.y*numCells.z+1;
    m_null_velgrid_value = numVelNodes.x*numVelNodes.y*numVelNodes.z+1;

    m_loopCounter = 0;
    m_loopCounterTotal = 0;

    m_worldOrigin = m_params.worldOrigin;
    m_numCells = numCells;
    m_numVelNodes = numVelNodes;

    m_minBin = m_params.minBin; //0.1;  //microns
    m_numBins = m_params.numBins;
    m_binSize = m_params.binSize; // microns
    printf("Starting a loop\n");
	//for (int n=1;n<m_numBins;n++) {
		 //m_params.binDiam[n] = n*bind;
		 //m_bVols[n] = 3.14159265359*powf((m_binSize*n/2.0e6),3.0) *4./3.; // # in cubic meters
	//}
	printf("Ending a loop\n");
	//bindiam[0] = smallest_diam;
	//m_bVols[0] = 3.14159265359*powf((m_minBin/2.0e6),3.0) *4./3.; // # in cubic meters


	printf("Allocating host storage\n");
    // allocate host storage
    printf("Allocate host storage (positions)\n");
    m_hPos          = new float[m_maxNumParticles*4];
    printf("Allocate host storage (sorted positions)\n");
    m_hSortedPos    = new float[m_maxNumParticles*4];
    printf("Allocate host storage (velocities)\n");
    m_hVel          = new float[m_maxNumParticles*4];
    printf("Allocate host storage (randoms)\n");
    m_hRandom       = new float[m_maxNumParticles*3];
    printf("Allocate host storage (scalars)\n");
    m_hScalar       = new float[m_maxNumParticles];
    printf("Allocate host storage (sorted scalars)\n");
    m_hSortedScalar = new float[m_maxNumParticles];
    printf("Allocate host storage (bins)\n");
    m_hDiamBins = new float[m_maxNumParticles*m_numBins];
    printf("Allocate host storage (sorted bins)\n");
    m_hSortedDiamBins = new float[m_maxNumParticles*m_numBins];
    printf("Allocate host storage (colors)\n");
    m_hColors = new float[m_maxNumParticles];

    // variables for velocity correction
    m_histLength = 1;
//    m_densHist = new float[m_numVelCells.x*m_numVelCells.z*m_histLength];
//    memset(m_densHist,0,m_histLength*m_numVelCells.x*m_numVelCells.z*sizeof(float));
//    m_dens = new float[m_numVelCells.x*m_numVelCells.z];
//    memset(m_dens,0,m_numVelCells.x*m_numVelCells.z*sizeof(float));
    m_densHist = new float[m_numCells.x*m_numCells.z*m_histLength];
    memset(m_densHist,0,m_histLength*m_numCells.x*m_numCells.z*sizeof(float));
    m_dens = new float[m_numCells.x*m_numCells.z];
    memset(m_dens,0,m_numCells.x*m_numCells.z*sizeof(float));
    //m_densCell = new float[m_numCells.x*m_numCells.z*m_histLength];
    //memset(m_densCell,0,m_histLength*m_numCells.x*m_numCells.z*sizeof(float));
    //m_densVelCell = new float[m_numVelCells.x*m_numVelCells.z*m_histLength];
    //memset(m_densVelCell,0,m_histLength*m_numVelCells.x*m_numVelCells.z*sizeof(float));
    //RMKrho
    //m_cellRho = new float[m_numCells.x*m_numCells.z];
    //memset(m_cellRho,0,m_numCells.x*m_numCells.z*sizeof(float));

    //m_UCorrect = new float[m_numCells.x*m_numCells.z*5];
    //memset(m_UCorrect,0,5*m_numCells.x*m_numCells.z*sizeof(float));
    n_iter = 0;

    printf("MEMSET (positions)\n");
    memset(m_hPos,          0, m_maxNumParticles*4*sizeof(float));
    printf("MEMSET (sorted positions)\n");
    memset(m_hSortedPos,    0, m_maxNumParticles*4*sizeof(float));
    printf("MEMSET (velocities)\n");
    memset(m_hVel,          0, m_maxNumParticles*4*sizeof(float));
    printf("MEMSET (randoms)\n");
    memset(m_hRandom,       0, m_maxNumParticles*sizeof(float));
    printf("MEMSET (scalars)\n");
    memset(m_hScalar,       0, m_maxNumParticles*sizeof(float));
    printf("MEMSET (sorted scalars)\n");
    memset(m_hSortedScalar, 0, m_maxNumParticles*sizeof(float));
    printf("MEMSET (bins)\n");
    memset(m_hDiamBins, 0, m_maxNumParticles*m_numBins*sizeof(float));
    printf("MEMSET (sorted bins)\n");
    memset(m_hSortedDiamBins, 0, m_maxNumParticles*m_numBins*sizeof(float));
    printf("MEMSET (colors)\n");
    memset(m_hColors, 0, m_maxNumParticles*sizeof(float));

    printf("MEMSET (cellstart)\n");
    m_hCellStart    = new uint[m_numTotalCells];
    m_hVelCellStart = new uint[m_numTotalVelNodes];
    m_hCellEnd      = new uint[m_numTotalCells];
    m_hVelCellEnd   = new uint[m_numTotalVelNodes];
    printf("MEMSET (cellend) %d %d\n",m_numTotalCells,m_numTotalVelNodes);
    memset(m_hCellStart,    0, m_numTotalCells   *sizeof(uint));
    memset(m_hVelCellStart, 0, m_numTotalVelNodes*sizeof(uint));
    memset(m_hCellEnd,      0, m_numTotalCells   *sizeof(uint));
    memset(m_hVelCellEnd,   0, m_numTotalVelNodes*sizeof(uint));

    // RMK: these are my creations
    printf("MEMSET (particlehash)\n");
    m_hParticleHash = new uint[m_maxNumParticles];
    memset(m_hParticleHash, 0, m_maxNumParticles*sizeof(uint));
    printf("MEMSET (particleindex)\n");
    m_hParticleIndex = new uint[m_maxNumParticles];
    memset(m_hParticleIndex, 0, m_maxNumParticles*sizeof(uint));
    printf("Allocate statistic storage\n");
    //m_cellVel = new float[m_numCells.x][m_numCells.z];
    m_cellMass = new float[m_numCells.x*m_numCells.z];
    m_cellScalar = new float[m_numCells.x*m_numCells.z];
    m_cellD10 = new float[m_numCells.x*m_numCells.z];
    m_cellParts = new uint[m_numCells.x*m_numCells.z];
    m_cellRMSScalar = new float[m_numCells.x*m_numCells.z];
    m_cellAvgScalar = new float[m_numCells.x*m_numCells.z];
    m_cellDiams = new float[m_numCells.x*m_numCells.z*m_numBins];
    memset(m_cellMass, 0, m_numCells.x*m_numCells.z*sizeof(float));
    memset(m_cellScalar, 0, m_numCells.x*m_numCells.z*sizeof(float));
    memset(m_cellD10, 0, m_numCells.x*m_numCells.z*sizeof(float));
    memset(m_cellParts, 0, m_numCells.x*m_numCells.z*sizeof(uint));
    memset(m_cellRMSScalar, 0, m_numCells.x*m_numCells.z*sizeof(float));
    memset(m_cellAvgScalar, 0, m_numCells.x*m_numCells.z*sizeof(float));
    memset(m_cellDiams, 0, m_numCells.x*m_numCells.z*m_numBins*sizeof(float));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_maxNumParticles;

    if (m_bUseOpenGL)
    {
    	printf("GL/VBO creation\n");
        m_posVbo = createVBO(memSize);
        printf("Registering buffer object\n");
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
        printf("Registered buffer object\n");
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    }

    printf("Allocating device arrays\n");
    allocateArray((void **)&m_dVel, memSize);
    allocateArray((void **)&m_dRandom, m_maxNumParticles*sizeof(float)*3);
    //allocateArray((void **)&m_dRandom1, m_maxNumParticles*sizeof(float));
    //allocateArray((void **)&m_dRandom2, m_maxNumParticles*sizeof(float));
    allocateArray((void **)&m_dScalar,  m_maxNumParticles*sizeof(float));
    allocateArray((void **)&m_dDiamBins,m_maxNumParticles*m_numBins*sizeof(float));
    allocateArray((void **)&m_dColors,  m_maxNumParticles*sizeof(float));

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
    allocateArray((void **)&m_dSortedScalar, m_maxNumParticles*sizeof(float));
    allocateArray((void **)&m_dSortedDiamBins, m_maxNumParticles*m_numBins*sizeof(float));

    allocateArray((void **)&m_dGridParticleHash, m_maxNumParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_maxNumParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart,    m_numTotalCells   *sizeof(uint));
    allocateArray((void **)&m_dVelCellStart, m_numTotalVelNodes*sizeof(uint));
    allocateArray((void **)&m_dCellEnd,      m_numTotalCells   *sizeof(uint));
    allocateArray((void **)&m_dVelCellEnd,   m_numTotalVelNodes*sizeof(uint));

    allocateArray((void **)&m_dUz,      m_numTotalVelNodes*sizeof(float));
    allocateArray((void **)&m_dUr,      m_numTotalVelNodes*sizeof(float));
    allocateArray((void **)&m_dEpsTKE,  m_numTotalVelNodes*sizeof(float));
    allocateArray((void **)&m_dNut,     m_numTotalVelNodes*sizeof(float));
    //allocateArray((void **)&m_dMassdt,  m_numTotalVelCells*sizeof(float));
    //allocateArray((void **)&m_dMassFlux,m_numTotalVelCells*sizeof(float));
    allocateArray((void **)&m_dGradZNut,m_numTotalVelNodes*sizeof(float));
    allocateArray((void **)&m_dGradRNut,m_numTotalVelNodes*sizeof(float));
    allocateArray((void **)&m_dDUz,     m_numTotalCells*sizeof(float));
    allocateArray((void **)&m_dDUr,     m_numTotalCells*sizeof(float));
	printf("Arrays allocated\n");

	printf("Do huz\n");
	copyArrayToDevice(m_dUz,      m_hUz,      0, sizeof(float) * m_numTotalVelNodes);
	threadSync();
	printf("Do hur\n");
	copyArrayToDevice(m_dUr,      m_hUr,      0, sizeof(float) * m_numTotalVelNodes);
	threadSync();
	printf("Do eps\n");
	copyArrayToDevice(m_dEpsTKE,  m_hEpsTKE,  0, sizeof(float) * m_numTotalVelNodes);
	threadSync();
	printf("Do nut\n");
	copyArrayToDevice(m_dNut,     m_hNut,     0, sizeof(float) * m_numTotalVelNodes);
	threadSync();
	printf("Do gz\n");
	copyArrayToDevice(m_dGradZNut,m_hGradZNut,0, sizeof(float) * m_numTotalVelNodes);
	threadSync();
	printf("Do gr\n");
	copyArrayToDevice(m_dGradRNut,m_hGradRNut,0, sizeof(float) * m_numTotalVelNodes);
	printf("This is probably where is all goes wrong %d\n",m_numTotalCells);
	threadSync();
	copyArrayToDevice(m_dDUz,     m_hDUz,     0, sizeof(float) * m_numTotalCells);
	copyArrayToDevice(m_dDUr,     m_hDUr,     0, sizeof(float) * m_numTotalCells);
	printf("How did I do?\n");
	//threadSync();

    if (m_bUseOpenGL)
    {
    	printf("we are using open GL\n");
        m_colorVBO = createVBO(m_maxNumParticles*4*sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i=0; i<m_maxNumParticles; i++)
        {
            float t = i / (float) m_maxNumParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
            //colorRamp(t, ptr);
            colorRamp(0.0, ptr);
            ptr+=3;
#endif
            // RMK: this is probably the radius declaration, so I'm hijacking this
            /*if (i<m_numParticles) {
            	*ptr++ = 1.0f;
            } else {
            	*ptr++ = -1.0f;
            }*/
            // or i'm wrong...
            *ptr++ = 1.0f;
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*maxNumParticles*4));
    }
    printf("done iwth that\n");
    //cin.ignore();

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);
    //setCellParameters(&m_cellparams);

    m_bInitialized = true;

    // Figure out what the cell density is (not vel cell)
    printf("Diving into cell densities\n");
    float total_mass, total_vol;
    float R1,R2,Z1,Z2;
    for (int i=0;i<m_numCells.x;i++) {
    	for (int k=0;k<m_numCells.z;k++) {
			total_mass = 0.0;
			total_vol = 0.0;
			for (int ii=0;ii<m_numVelNodes.x-1;ii++) {
				for (int kk=0;kk<m_numVelNodes.z-1;kk++) {
					// check for R overlap
					if((m_R[ii+1]<=(i+1)*m_params.cellSize.x && m_R[ii+1]>=i*m_params.cellSize.x ||
						(i+1)*m_params.cellSize.x>=m_R[ii] && (i+1)*m_params.cellSize.x<=m_R[ii+1]) &&
//						m_R[ii]  <(i+1)*m_params.cellSize.x && m_R[ii]  >i*m_params.cellSize.x) &&
					   (m_Z[kk+1]<=(k+1)*m_params.cellSize.z && m_Z[kk+1]>=k*m_params.cellSize.z ||
						(k+1)*m_params.cellSize.z>=m_Z[kk] && (k+1)*m_params.cellSize.z<=m_Z[kk+1])) {
//						m_Z[kk]  <(k+1)*m_params.cellSize.z && m_Z[kk]  >k*m_params.cellSize.z)) {
						R1 = max(m_R[ii],   i   *m_params.cellSize.x);
						R2 = min(m_R[ii+1],(i+1)*m_params.cellSize.x);
						Z1 = max(m_Z[kk],   k   *m_params.cellSize.z);
						Z2 = min(m_Z[kk+1],(k+1)*m_params.cellSize.z);
						total_mass += (Z2-Z1) * m_params.darc/2.0*(powf(R2,2.)-powf(R1,2.)) * m_hRho[ii*m_numVelNodes.z+kk];
						total_vol  += (Z2-Z1) * m_params.darc/2.0*(powf(R2,2.)-powf(R1,2.));
						//printf("RZ: %f %f %f %f %f %f\n",R1,R2,Z1,Z2,total_mass,total_vol);
					}
				}
			}
			m_hRhoCells[i*m_numCells.z+k] = total_mass/total_vol;
			printf("Compare %e to %e / volume %e to %e (%d,%d)\n",m_hRhoCells[i*m_numCells.z+k],total_mass,total_vol,m_params.cellSize.z * m_darc/2.0 * (powf(m_params.cellSize.x*(i+1),2.0)-powf(m_params.cellSize.x*i,2.0)),i,k);
    	}
    	//cin.ignore();
    }

    printf("Exiting _initialize\n");

}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hSortedPos;
    delete [] m_hVel;
    delete [] m_hRandom;
    delete [] m_hScalar;
    delete [] m_hSortedScalar;
    delete [] m_hCellStart;
    delete [] m_hVelCellStart;
    delete [] m_hCellEnd;
    delete [] m_hVelCellEnd;
    delete [] m_hColors;

    delete [] m_hDiamBins;
    delete [] m_hSortedDiamBins;

    //delete [] m_cellVel;
    delete [] m_cellMass;
    delete [] m_cellD10;
    delete [] m_cellScalar;
    delete [] m_cellParts;
    delete [] m_cellAvgScalar;
    delete [] m_cellDiams;

    delete [] m_dens;
    delete [] m_densHist;

    freeArray(m_dVel);
    freeArray(m_dRandom);
    //freeArray(m_dRandom1);
    //freeArray(m_dRandom2);
    freeArray(m_dScalar);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
    freeArray(m_dSortedScalar);

    freeArray(m_dDiamBins);
    freeArray(m_dSortedDiamBins);
    freeArray(m_dColors);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dVelCellStart);
    freeArray(m_dCellEnd);
    freeArray(m_dVelCellEnd);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}
/*
// calculate position in non-uniform grid
int ParticleSystem::calcNewGridPos(float x, float y, float z)
{
    int gridPosx;
    int gridPosy;
    int gridPosz;

    gridPosx = 0;
    gridPosy = 0;
    gridPosz = 0;
    for (int i=0;i<=m_params.gridSize.x;i++) {
    	if (m_params.R[i+1]>=x) {
    		gridPosx = i;
    		break;
    	}
    }
    for (int i=0;i<=m_params.gridSize.z;i++) {
    	if (m_params.Z[i+1]>=z) {
    		gridPosz = i;
    		break;
    	}
    }
    return calcGridHashHost(gridPosx,gridPosy,gridPosz);
}
*/
// calculate address in grid from position (clamping to edges)

int ParticleSystem::calcVelGridHashHost(int gridPosx,int gridPosy,int gridPosz)
{
	// RMK: Better hope that gridPos !> gridSize-1
    //gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    //gridPos.y = gridPos.y & (params.gridSize.y-1);
    //gridPos.z = gridPos.z & (params.gridSize.z-1);
    return gridPosz * m_params.numVelNodes.y * m_params.numVelNodes.x + gridPosy * m_params.numVelNodes.x + gridPosx;
}

int ParticleSystem::calcGridHashHost(int gridPosx,int gridPosy,int gridPosz)
{
	// RMK: Better hope that gridPos !> gridSize-1
    //gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    //gridPos.y = gridPos.y & (params.gridSize.y-1);
    //gridPos.z = gridPos.z & (params.gridSize.z-1);
    return gridPosz * m_params.numCells.y * m_params.numCells.x + gridPosy * m_params.numCells.x + gridPosx;
}

void ParticleSystem::boundaryFluxHybrid(//float *newVel,            // calculate and return velocities
			 //float *newPos,
			 //float *newScalar,
			 //float *newDiamBins,
             float *sortedPos,         // accept the positions (sorted)
             float *sortedScalar,         // accept the positions (sorted)
             float *sortedDiamBins,
             uint  *gridParticleHash,
             uint  *gridParticleIndex,
             uint  *cellStart,
             uint  *cellEnd,
             uint   maxNumParticles,
             uint   &numParticles,
             float *cellVelDensity)
             //uint   numTotCells)
{

	int index = 0;
	//int j;
	float mass, cellMass;
	float total_mass = 0.0;
	int gindex;
	int cell;
	int lessParticles = 0;
	//float newRad = 0.1;
	printf("np: %d\n",numParticles);
	float fluxin = 0.0;
	float fluxout = 0.0;
	float totalflux = 0.0;
	float epsi = 1.0e-8;

	for (int i=0;i<m_params.numCells.x;i++) {
		for (int k=0;k<m_params.numCells.z;k++) {
			gindex = calcGridHashHost(i,0,k);
			//printf("ikg: %d %d %d",i,k,gindex);
			cellMass = 0.0;
			for (int b=cellStart[gindex];b<cellEnd[gindex];b++) {
				cellMass += sortedPos[b*4+3];
			}
			//printf(" past bs\n");
//			cellVelDensity[gindex] = cellMass/cellVelVolume(make_uint3(i,0,k));
			total_mass += cellMass;
			//printf("%d %d %d\n",i,k,gindex);
			if (i==m_params.numCells.x-1 || k==0 || k==m_params.numCells.z-1) {
				//printf("weeee!\n");
				totalflux += m_hMassdt[i*m_params.numCells.z+k];
				if (m_hMassdt[i*m_params.numCells.z+k]>0) {  // add a particle
					//printf("Adding particle! %d %d\n",i,k);
					fluxin += m_hMassdt[i*m_params.numCells.z+k];
					m_hMassFlux[i*m_params.numCells.z+k] -= m_hMassdt[i*m_params.numCells.z+k];
					//while ((m_massdt[i][k]/fabs(m_massdt[i][k]))*m_massflux[i][k] < 0.0) {
					while (m_hMassFlux[i*m_params.numCells.z+k] < 0.0) {
						//if (i==m_params.numVelCells.x-1) {
						//	printf("Adding part, %f\n",sortedPos[0]);
						//}
						//printf("in this while, right? %d %d %d %d %e %e\n",index, m_hMassFlux[i*m_params.numCells.z+k]);
						//cin.ignore();
						if (k==0) { // adding from bottom of domain
							sortedPos[numParticles*4  ] = (i+0.5)*m_params.cellSize.x; //(m_R[i+1]-m_R[i]) * frand() + m_R[i]; //p.x;
						} else { // adding from side of domain
							sortedPos[numParticles*4  ] = (i+1)*m_params.cellSize.x-epsi; //m_params.cellSize.x*(m_params.numCells.x-1) + frand()*m_params.cellSize.x; //p.z;
						}
						//sortedPos[numParticles*4  ] = (m_R[i+1]-m_R[i]) * frand() + m_R[i]; //p.x;
						//printf("Adding part 2\n");
						sortedPos[numParticles*4+1] = 0.0; //p.y;
						//printf("Adding part 3\n");
						//sortedPos[numParticles*4+2] = (m_Z[k+1]-m_Z[k]) * frand() + m_Z[k]; //p.z;
						if (k==0) { // adding from bottom of domain
							sortedPos[numParticles*4+2] = k*m_params.cellSize.z+epsi; //frand() * m_params.cellSize.z;
						} else { // adding from side of domain
							sortedPos[numParticles*4+2] = (k+frand())*m_params.cellSize.z; //p.z;
						}
						//printf("Adding part 4\n");
						sortedPos[numParticles*4+3] = cellVolume(calcGridPos(sortedPos[numParticles*4],0.,sortedPos[numParticles*4+2]))*interpRho(sortedPos[numParticles*4],sortedPos[numParticles*4+2])/m_params.avgParticlesPerCell; //0.01;
						//sortedPos[numParticles*4+3] = cellVelVolume(make_uint3(i,0,k))*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
						m_hMassFlux[i*m_params.numCells.z+k] += sortedPos[numParticles*4+3];
						m_mass_added += sortedPos[numParticles*4+3];
						//printf("Adding part 5\n");
						//if ((m_R[i+1]+m_R[i])/2.0<m_params.jet_radius && k==0) {
						for (int b=0; b<m_params.numBins; b++) {
							sortedDiamBins[numParticles*m_params.numBins+b] = 0.0;
						}
						if ((i+0.5)*m_params.cellSize.x<m_params.jet_radius && k==0) {
							sortedScalar[numParticles] = 1.0;
							//printf("injected a particle with scalar!\n");
							if (not m_params.SEEDED) {
								sortedDiamBins[numParticles*m_params.numBins] = m_params.jet_conc_nuc*cellVolume(make_uint3(i,0,k))/m_params.avgParticlesPerCell*1.0e6;  // 1e6 needed to convert concentration from cm^-3 to m^-3
							} else {
								sortedDiamBins[numParticles*m_params.numBins+m_params.seededBin] = m_params.jet_conc_nuc*cellVolume(make_uint3(i,0,k))/m_params.avgParticlesPerCell*1.0e6;  // 1e6 needed to convert concentration from cm^-3 to m^-3
								sortedPos[numParticles*4+3] += (m_params.bVols[m_params.seededBin]-m_params.bVols[0]) * m_params.density_h2o * sortedDiamBins[numParticles*m_params.numBins+m_params.seededBin];
							}
							//printf("m_R[i+1] %f %f\n",m_R[i+1],m_params.jet_radius);
						} else {
							sortedScalar[numParticles] = 0.0; //5;
							sortedDiamBins[numParticles*m_params.numBins] = m_params.conc_nuc_vol*cellVolume(make_uint3(i,0,k))/m_params.avgParticlesPerCell*1.0e6;  // 1e6 needed to convert concentration from cm^-3 to m^-3
						}
						numParticles += 1;
					}
				} else { // remove particles
					//printf("Removing particle! %d %d\n",i,k);
					fluxout += m_hMassdt[i*m_params.numCells.z+k];
					/*
					mass = 0.0;
					for (int b=cellStart[gindex];b<cellEnd[gindex];b++) {
						mass += sortedPos[b*4+3];
					}*/
					//printf("Check cell exit: %d %d %12.10f %12.10f %12.10f v %12.10f\n",i,k,mass,cellVelVolume(make_uint3(i,0,k)),mass/cellVelVolume(make_uint3(i,0,k)),m_params.celldensity);
					// We need to remove particles very deterministically now that we are making adjustments to the velocities...
					m_hMassFlux[i*m_params.numCells.z+k] += m_hMassdt[i*m_params.numCells.z+k];
					//printf("going WHILE\n");
					while (m_hMassFlux[i*m_params.numCells.z+k] < 0.0) {
					//while (mass/cellVelVolume(make_uint3(i,0,k))>m_params.celldensity) {
						index = cellStart[gindex];
						cellStart[gindex] += 1;
						if (index>cellEnd[gindex]-1) {
							//saveCells();
							//printf("WE GOT A HUGE PROBLEM!.......%d...%d.......\n",i,k);
							break;
							//cin.ignore();
						} else {
							//printf("in this else, right? %d %d %d %d %e %e\n",index,gindex,cellStart[gindex],cellEnd[gindex],sortedPos[index*4+3], m_hMassFlux[i*m_params.numCells.z+k]);
							//cin.ignore();
							m_hMassFlux[i*m_params.numCells.z+k] += sortedPos[index*4+3];
							m_mass_removed += sortedPos[index*4+3];
							//mass = mass - sortedPos[index*4+3];
							//m_params.massflux[i][k] -= sortedPos[index*4+3];
							sortedPos[index*4+3] = -1.0;
							lessParticles += 1;
						}
					}
				}
			}
		}
	}

	printf("Changing NP bf from %d to %d\n",numParticles,numParticles-lessParticles);
	//cin.ignore();
	printf("Total mass in the domain: %f\n",total_mass);
	printf("Because in %e out %e total %e\n",fluxin,fluxout,totalflux);
	printf("Added %e vs. removed %e\n",m_mass_added,m_mass_removed);

	numParticles -= lessParticles;

}

// CHECK OUT THE INDEXING
void ParticleSystem::mixCurl(            // calculate and return velocities
		 	 //float *newPos,
		 	 //float *newScalar,
		 	 //float *newDiamBins,
		 	 float *sortedPos,         // accept the positions (sorted)
		 	 float *sortedScalar,         // accept the positions (sorted)
        	 float *sortedDiamBins,
             //uint  *gridParticleHash,
             //uint  *gridParticleIndex,
             uint  *cellStart,
             uint  *cellEnd,
             float *densCell)
             //uint   maxNumParticles,
             //uint   &numParticles)
             //uint   numTotCells)
{
	int npart;
	int a, b;
	int gindex;
	float avg_scalar;
	//float* avg_bins;
	float* bins_mixed;
	float rannum;
	bins_mixed = new float[m_params.numBins];
	float cellMass;
	float num_mix_pairs,mixing_degree,mass_mixed;

	for (int k=0;k<m_numCells.z;k++) {
		for (int i=0;i<m_numCells.x;i++) {

			gindex = calcGridHashHost(i,0,k);

			cellMass = 0.0;
			for (int j=cellStart[gindex];j<cellEnd[gindex];j++) {
				cellMass += sortedPos[j*4+3];
				//printf("%d %f %d %d\n",gindex,sortedPos[j*4+3],cellStart[gindex],cellEnd[gindex]);
			}
			densCell[gindex] = cellMass/cellVolume(make_uint3(i,0,k));
			//printf("density in cell: %f %f %f %d %d\n",densCell[gindex],cellMass,cellVolume(make_uint3(i,0,k)),i,k);


			rannum = frand();
			npart = cellEnd[gindex]-cellStart[gindex];
			//OLDER CODE DEBUG:
			//if (rannum < m_params.Cphi*npart*m_hEpsTKE[i*m_params.numVelNodes.z+k]*m_params.dt && npart>1) {
			//if (rannum < m_params.Cphi*npart*interpEpsTKE((i+0.5)*m_params.cellSize.x,(k+0.5)*m_params.cellSize.z)*m_params.dt && npart>1) {
			//if (rannum < m_params.Cphi*npart*max(100.0f,interpEpsTKE((i+0.5)*m_params.cellSize.x,(k+0.5)*m_params.cellSize.z))*m_params.dt && npart>1) {
			//if (false) {
			num_mix_pairs = m_params.Cphi*npart*interpEpsTKE((i+0.5)*m_params.cellSize.x,(k+0.5)*m_params.cellSize.z)*m_params.dt;
			if (num_mix_pairs>npart/2.0) {
				printf("The mixing vigor exceeds the particle capacity %f > %d\n",num_mix_pairs,npart);
			}
			while (num_mix_pairs>0.0) {
				// Determine if we keep mixing; if so, that's one less pair to mix
				if (num_mix_pairs>1.0) {
					num_mix_pairs = num_mix_pairs - 1;
				} else {
					if (frand()>num_mix_pairs) {
						break;
					} else {
						num_mix_pairs = -1.0;
					}
				}
				// Randomly pick which two particles will mix
				//a = random.randint(0,cells[i][k].np-1);
				a = floor(frand()*npart);
				//b = a;
				b = floor(frand()*npart);
				while (a == b) {
					//b = random.randint(0,cells[i][k].np-1)
					b = floor(frand()*npart);
				}
				a = cellStart[gindex]+a;
				b = cellStart[gindex]+b;

				// Randomly determine how much mixing will occur
				mixing_degree = frand();

				//float conc_a = sortedScalar[a]*(m_params.jet_conc_nuc-m_params.conc_nuc_vol)+m_params.conc_nuc_vol;
				//float conc_b = sortedScalar[b]*(m_params.jet_conc_nuc-m_params.conc_nuc_vol)+m_params.conc_nuc_vol;

				// Calculate scalar in mixed region
				if ((sortedPos[a*4+3]+sortedPos[b*4+3])>0.0) {
					avg_scalar = (sortedScalar[a]*sortedPos[a*4+3] +
								  sortedScalar[b]*sortedPos[b*4+3]) / (sortedPos[a*4+3]+sortedPos[b*4+3]);
				} else {
					// This declarations shouldn't matter too much since both particle masses are zero
					avg_scalar = (sortedScalar[a] + sortedScalar[b]) / 2.0;
				}

				//float conc_avg = avg_scalar*(m_params.jet_conc_nuc-m_params.conc_nuc_vol)+m_params.conc_nuc_vol;

				if (sortedPos[a*4+3]>0.0 || sortedPos[b*4+3]>0.0) {
					mass_mixed = (sortedPos[a*4+3]+sortedPos[b*4+3])*mixing_degree;
					for (int bb=0;bb<m_params.numBins;bb++) {
						bins_mixed[bb] = (sortedDiamBins[a*m_params.numBins+bb]+sortedDiamBins[b*m_params.numBins+bb])*mixing_degree;
					}
					sortedScalar[a] = (sortedScalar[a]*sortedPos[a*4+3]*(1.0-mixing_degree) +
									   avg_scalar     *mass_mixed/2.0)/(sortedPos[a*4+3]*(1.0-mixing_degree)+mass_mixed/2.0);
					sortedScalar[b] = (sortedScalar[b]*sortedPos[b*4+3]*(1.0-mixing_degree) +
									   avg_scalar     *mass_mixed/2.0)/(sortedPos[b*4+3]*(1.0-mixing_degree)+mass_mixed/2.0);
					sortedPos[a*4+3] = sortedPos[a*4+3]*(1.0-mixing_degree) + mass_mixed/2.0;
					sortedPos[b*4+3] = sortedPos[b*4+3]*(1.0-mixing_degree) + mass_mixed/2.0;
					for (int bb=0;bb<m_params.numBins;bb++) {
						sortedDiamBins[a*m_params.numBins+bb] = sortedDiamBins[a*m_params.numBins+bb]*(1.0-mixing_degree) + bins_mixed[bb]/2.0;
						sortedDiamBins[b*m_params.numBins+bb] = sortedDiamBins[b*m_params.numBins+bb]*(1.0-mixing_degree) + bins_mixed[bb]/2.0;
					}
				} else {
					// These declarations shouldn't matter too much since both particle masses are zero
					sortedScalar[a] = avg_scalar;
					sortedScalar[b] = avg_scalar;
					for (int bb=0;bb<m_params.numBins;bb++) {
						sortedDiamBins[a*m_params.numBins+bb] = 0.0;
						sortedDiamBins[b*m_params.numBins+bb] = 0.0;
					}
					printf("Mixing in name only\n");
				}
			//} else if (npart < 2) {
			//	printf("er, there are not enough particles to mix %d,%d\n",i,k);
			}
		}
	}
	//cin.ignore();
	delete [] bins_mixed;
}

void ParticleSystem::balanceCells(            // calculate and return velocities
			 //float *newPos,
			 //float *newScalar,
             float *sortedPos,         // accept the positions (sorted)
			 float *sortedScalar,
			 float *sortedDiamBins,
             uint  *gridParticleHash,
             uint  *gridParticleIndex,
             uint  *cellStart,
             uint  *cellEnd,
             uint   maxNumParticles,
             uint   &numParticles)
             //uint   numTotCells)
{
	int index = 0;
	int gindex;
	int cell;
	int lessParticles = 0;
	//float newRad = 0.1;
	printf("npbc: %d\n",numParticles);
	int partsInCell=-1;
	int j;
	int total = 0;

	for (int i=0;i<m_params.numCells.x;i++) {
		for (int k=0;k<m_params.numCells.z;k++) {
			gindex = calcGridHashHost(i,0,k);
			//printf("cs %d %d\n",cellStart[gindex],numCells);
			//partsInCell = -1;
			if (cellStart[gindex]!=-1) {
				//printf("Yay\n");
				partsInCell = cellEnd[gindex]-cellStart[gindex];
				//printf("parts in cell %d %d %d %d %d %d\n",partsInCell,cellEnd[gindex],cellStart[gindex],i,k,gindex);
				j=0;
			} //else {
			//printf("parts in cell %d %d %d %d %d %d\n",partsInCell,cellEnd[gindex],cellStart[gindex],i,k,gindex);
			//cin.ignore();
			if (partsInCell<=0) {
				//printf("nay\n");
				//printf("No particles in cell!\n");
				//cin.ignore();
				partsInCell = 0;
				cellStart[gindex] = numParticles;
				cellEnd[gindex] = numParticles;
				//sortedPos[numParticles*4  ] = (m_R[i+1]-m_R[i]) * frand() + m_R[i];
				sortedPos[numParticles*4  ] = m_params.worldSize.x * frand() + m_worldOrigin.x;
				sortedPos[numParticles*4+1] = 0.0; //p.y;
				//sortedPos[numParticles*4+2] = (m_Z[k+1]-m_Z[k]) * frand() + m_Z[k]; //p.z;
				sortedPos[numParticles*4+2] = m_params.worldSize.z * frand() + m_worldOrigin.z;
				sortedPos[numParticles*4+3] = 0.0; //volume(gindex)*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
				sortedScalar[numParticles] = 0.0;
				//sortedDiamBins[numParticles*m_params.numBins] = 1.0;
				sortedDiamBins[numParticles*m_params.numBins] = 0.0;
				for (int b=1; b<m_params.numBins; b++) {
					sortedDiamBins[numParticles*m_params.numBins+b] = 0.0;
				}
				numParticles++;
				//partsInCell = 1;
				total++;
				j=1;
			}
			//j = 0;
			//printf("pic: %d %d %d %d %d %d\n",cellStart[gindex],cellEnd[gindex],partsInCell,i,k,gindex);
			while (partsInCell+j<m_params.minParticlesPerCell) {
				//printf("check1\n");
				sortedPos[numParticles*4  ] = (m_R[i+1]-m_R[i]) * frand() + m_R[i];
				//printf("check2\n");
				sortedPos[numParticles*4+1] = 0.0; //p.y;
				//printf("check3\n");
				sortedPos[numParticles*4+2] = (m_Z[k+1]-m_Z[k]) * frand() + m_Z[k]; //p.z;
				//printf("check4\n");
				sortedPos[numParticles*4+3] = 0.0; //volume(gindex)*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
				sortedScalar[numParticles] = 0.0;
				if (j<partsInCell) {
					sortedPos   [numParticles         *4+3] = sortedPos   [(cellStart[gindex]+j)*4+3]/2.0;
					if (sortedPos[numParticles*4+3]<0) {printf("what? %d %d %d\n",cellStart[gindex],cellEnd[gindex],j);}
					sortedScalar[numParticles             ] = sortedScalar[ cellStart[gindex]+j]; ///2.0;
					sortedPos   [(cellStart[gindex]+j)*4+3] = sortedPos   [numParticles*4+3];
					//sortedDiamBins[numParticles*m_params.numBins] = 1.0;
					for (int b=0; b<m_params.numBins; b++) {
						sortedDiamBins[numParticles*m_params.numBins+b] = sortedDiamBins[(cellStart[gindex]+j)*m_params.numBins+b]/2.0;
						sortedDiamBins[(cellStart[gindex]+j)*m_params.numBins+b] = sortedDiamBins[numParticles*m_params.numBins+b];
					}
					//sortedScalar[cellStart[gindex]+j] = sortedScalar[numParticles];
					if (sortedPos   [numParticles         *4+3]<0) {
					printf("new cell mass: %f %d %d %d %d %f %d %d\n",sortedPos[numParticles*4+3],j, gindex, cellStart[gindex], cellEnd[gindex],sortedPos[(cellStart[gindex]+j)*4+3],(cellStart[gindex]+j),m_hCellStart[gindex]);
					}
				} else {
					sortedPos   [numParticles*4+3] = 0.0;
					sortedScalar[numParticles    ] = 0.0;
					sortedDiamBins[numParticles*m_params.numBins] = 0.0;
					for (int b=1; b<m_params.numBins; b++) {
						sortedDiamBins[numParticles*m_params.numBins+b] = 0.0;
					}
				}
				//printf("check5\n");

				numParticles += 1;
				//partsInCell += 1;
				j++;
				total++;
				//printf("check6\n");
			}
			// For now, we're assuming we never end up with twice (or more) as many particles as we need
			while (partsInCell>m_params.maxParticlesPerCell) {
				//printf("check7\n");
				printf("parts: %d, (%d,%d)\n",partsInCell,i,k);
				if ((sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3])>0.0) {
					sortedScalar[cellStart[gindex]+j] = (sortedScalar[cellStart[gindex]+j]*sortedPos[(cellStart[gindex]+j)*4+3] + sortedScalar[cellEnd[gindex]-1]*sortedPos[(cellEnd[gindex]-1)*4+3])
													    / (sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3]);
					sortedPos[(cellStart[gindex]+j)*4  ] = (sortedPos[(cellStart[gindex]+j)*4  ]*sortedPos[(cellStart[gindex]+j)*4+3] + sortedPos[(cellEnd[gindex]-1)*4  ]*sortedPos[(cellEnd[gindex]-1)*4+3])
														/ (sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3]);
					sortedPos[(cellStart[gindex]+j)*4+2] = (sortedPos[(cellStart[gindex]+j)*4+2]*sortedPos[(cellStart[gindex]+j)*4+3] + sortedPos[(cellEnd[gindex]-1)*4+2]*sortedPos[(cellEnd[gindex]-1)*4+3])
														/ (sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3]);
					sortedScalar[cellStart[gindex]+j] = (sortedScalar[cellStart[gindex]+j]*sortedPos[(cellStart[gindex]+j)*4+3] + sortedScalar[cellEnd[gindex]-1]*sortedPos[(cellEnd[gindex]-1)*4+3])
														/ (sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3]);
					for (int b=0; b<m_params.numBins; b++) {
						sortedDiamBins[(cellStart[gindex]+j)*m_params.numBins+b] =  sortedDiamBins[(cellStart[gindex]+j)*m_params.numBins+b] + sortedDiamBins[(cellEnd[gindex]-1)*m_params.numBins+b];
					}
					sortedPos[(cellStart[gindex]+j)*4+3] = sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3];
				} else {
					sortedScalar[cellStart[gindex]+j] = 0.0;
				}
/*				sortedPos[(cellStart[gindex]+j)*4+3] += sortedPos[cellEnd[gindex]*4+3];
*/				sortedPos[(cellEnd[gindex]-1)*4+3] = -1.0;
				cellEnd[gindex] -= 1;
				lessParticles += 1;
				partsInCell -= 1;
				j++;
				total--;
			}
			//cin.ignore();
			//total += partsInCell;
		}
	}

	printf("Changing NP from %d to %d vs. %d\n",numParticles,numParticles-lessParticles,total);

	numParticles -= lessParticles;

/*	uint originalIndex;
	for (int i=0;i<maxNumParticles;i++) {
	    // write new velocity back to original unsorted location
	    originalIndex = gridParticleIndex[i];
	    //newVel[originalIndex] = make_float4(vel + force, 0.0f);
	    newPos[originalIndex*4  ] = sortedPos[i*4  ];
	    newPos[originalIndex*4+1] = sortedPos[i*4+1];
	    newPos[originalIndex*4+2] = sortedPos[i*4+2];
	    newPos[originalIndex*4+3] = sortedPos[i*4+3];
	    newScalar[originalIndex]  = sortedScalar[i];
	}
*/
}

// function for Antoine's equation; returns pressure in Pa, given temp in C
//float AE(float temperature) {
//	const float AE_A = 8.05573;
//	const float AE_B = 1723.64;
//	const float AE_C = 233.076;

//	return powf(10.0,(AE_A-AE_B/(temperature+AE_C))) * 133.32239;
//}

// function for Antoine's equation; returns pressure in mm Hg, given temp in C
float AE_mmHg2(float temperature) {
	const float AE_A = 8.05573;
	const float AE_B = 1723.64;
	const float AE_C = 233.076;

	return powf(10.0,(AE_A-AE_B/(temperature+AE_C)));
}
// no idea if this is right
float AEinv2(float pressure) {
	const float AE_A = 8.05573;
	const float AE_B = 1723.64;
	const float AE_C = 233.076;

	return -AE_B/(log10(pressure / 133.32239)-AE_A) - AE_C;
}
float ParticleSystem::scale1toC(float temperature) {
	return temperature*(m_params.tj-m_params.t0)+m_params.t0;
}
float ParticleSystem::scaleCto1(float temperature) {
	return (temperature-m_params.t0)/(m_params.tj-m_params.t0);
}
float ParticleSystem::scale1toPa(float humidity) {
	return humidity*(m_params.ppwj-m_params.ppw0)+m_params.ppw0;
}
float ParticleSystem::scalePato1(float humidity) {
	return (humidity-m_params.ppw0)/(m_params.ppwj-m_params.ppw0);
}
float ParticleSystem::w2p(float w) {
	return w*m_params.patm/(w+1);
}
float ParticleSystem::ss_calc(float scalar,float *radii,float *bins) {
	float delta_press = 0.0;
	float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	for(int b=0;b<m_params.numBins;b++) {
		delta_temp  += m_params.lambda_h2o/m_params.Cp * ((4.0/3.0*3.14159265359*powf(radii[b],3.0)-m_params.bVols[0]) * m_params.density_h2o/m_params.density_air) * bins[b]*1.0e6;
		delta_press += -w2p( (4.0/3.0*3.14159265359*powf(radii[b],3.0)-m_params.bVols[0]) * m_params.density_h2o/m_params.density_air) * bins[b]*1.0e6;
	}
	return (scale1toPa(scalar)+delta_press)/AE(scale1toC(scalar)+delta_temp);
}
float ParticleSystem::ss_calck(float scalar,float *radii,float *bins,float *k,float dt) {
	float delta_press = 0.0;
	float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	for(int b=0;b<m_params.numBins;b++) {
		delta_temp  += m_params.lambda_h2o/m_params.Cp * ((4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-m_params.bVols[0]) * m_params.density_h2o/m_params.density_air) * bins[b]*1.0e6;
		delta_press += -w2p( (4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-m_params.bVols[0]) * m_params.density_h2o/m_params.density_air) * bins[b]*1.0e6;
	}
	//printf("deltas: %f %f %f %f %f\n",scale1toC(scalar),delta_temp,AE(scale1toC(scalar)+delta_temp),scale1toPa(scalar),delta_press);
	return (scale1toPa(scalar)+delta_press)/AE(scale1toC(scalar)+delta_temp);
}
float ParticleSystem::ss_calc_enthalpy(float scalar,float *radii,float *bins,float particle_mass) {
	//float delta_press = 0.0;
	//float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	float X = (1.0-scalar)*m_params.x0 + scalar*m_params.xj;  // mass total water / total particle mass
	float hp = (1.0-scalar)*m_params.h0 + scalar*m_params.hj; // calculate enthalpy of particle
	float Xs = 0.0;
	for (int b=0;b<m_params.numBins;b++) {
		Xs += (4.0/3.0*3.14159265359*powf(radii[b],3.0)-m_params.bVols[0]) * m_params.density_h2o * bins[b]; //*1.0e6;
	}
	Xs = Xs/particle_mass;
	//Xs = 0.0;
	float T = (hp - (X-Xs)*m_params.hwe)/((1.0-X)*m_params.Cp+(X-Xs)*m_params.Cpw+Xs*m_params.Cw);
	float Ps = 760.0*(1.0 - (1.0-X)/((1.0-X)+(X-Xs)*m_params.Mair/m_params.Mwater));
	//printf("%f, %f, %f, %f, %f\n",Ps,T,Ps/AE_mmHg(T),X,Xs);
	for (int b=0;b<m_params.numBins;b++) {
		if (radii[b]>10.0e-6 && bins[b]>0.0) {
			printf("so TIRED.... %d  %e : %e X %e,%e,%e  %e,%e,%e,%e\n",b,radii[b],m_params.bDiams[b]/2e6,X,1.-X,Xs,(hp - (X)*m_params.hwe)/((1.0-X)*m_params.Cp+(X)*m_params.Cpw),T,Ps/AE_mmHg2(T),bins[b]);
		}
	}
	return Ps/AE_mmHg2(T);
}
float ParticleSystem::ss_calck_enthalpy(float scalar,float *radii,float *bins,float *k,float dt,float particle_mass) {
	float X = (1.0-scalar)*m_params.x0 + scalar*m_params.xj;  // mass total water / total particle mass
	float hp = (1.0-scalar)*m_params.h0 + scalar*m_params.hj; // calculate enthalpy of particle
	float Xs = 0.0;
	for (int b=0;b<m_params.numBins;b++) {
		Xs += (4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-m_params.bVols[0]) * m_params.density_h2o * bins[b]; //*1.0e6;
	}
	Xs = Xs/particle_mass;
	//Xs = 0.0;
	float T = (hp - (X-Xs)*m_params.hwe)/((1.0-X)*m_params.Cp+(X-Xs)*m_params.Cpw+Xs*m_params.Cw);
	float Ps = 760.0*(1.0 - (1.0-X)/((1.0-X)+(X-Xs)*m_params.Mair/m_params.Mwater));
	return Ps/AE_mmHg2(T);
}
void ParticleSystem::drdt_bins(float scalar,float *bins,float dt,float *radii,float particle_mass) {
	float k1[NUM_DIAM_BINS];
	float k2[NUM_DIAM_BINS];
	float k3[NUM_DIAM_BINS];
	float k4[NUM_DIAM_BINS];
	float gammaOverRadius[NUM_DIAM_BINS];

	int bfirst;

	for (int b=0;b<m_params.numBins;b++) {
		gammaOverRadius[b] = m_params.gamma / radii[b];
	}

	float ss = ss_calc_enthalpy(scalar,radii,bins,particle_mass)-1.0;
	if (ss>0.0) { bfirst = 0; } else { bfirst = 1; k1[0] = 0.0; k2[0] = 0.0; k3[0] = 0.0; k4[0] = 0.0; }
	for (int b=bfirst;b<m_params.numBins;b++) {
		k1[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k1,dt/2.0,particle_mass)-1.0;
	for (int b=bfirst;b<m_params.numBins;b++) {
		k2[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k2,dt/2.0,particle_mass)-1.0;
	for (int b=bfirst;b<m_params.numBins;b++) {
		k3[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k3,dt    ,particle_mass)-1.0;
	for (int b=bfirst;b<m_params.numBins;b++) {
		k4[b] = ss * gammaOverRadius[b];
	}

	for (int b=0;b<m_params.numBins;b++) {
		radii[b] = max(radii[b] + (k1[b]+2*k2[b]+2*k3[b]+k4[b])*dt/6.0,m_params.minBin/2e6);
		if (radii[b]>20.0e-6 && bins[b]>0.0) {
			printf("so tired.... %d  %e : %e ks %e,%e,%e,%e\n",b,radii[b],m_params.bDiams[b]/2e6,k1[b],k2[b],k3[b],k4[b]);
		}
	}
}

void ParticleSystem::RK4(float scalar,float *bins,float dt,float particle_mass) {
	const int N = 10;
	float ddt = dt/N;
	float radii[NUM_DIAM_BINS];
	float newBins[NUM_DIAM_BINS];
	const float threshold = 1e-4/NUM_DIAM_BINS;

    for (int b=0;b<m_params.numBins;b++) {
		radii[b] = m_params.bDiams[b]/2.0e6;
		newBins[b] = 0.0;
	}
	// Break up dt into smaller timesteps for integrating
	for (int n=0;n<N;n++) {
		drdt_bins(scalar,bins,ddt,radii,particle_mass);
	}

	float Vnew;
	int b1, b2;
	float y1, y2;
	for (int b=0;b<m_params.numBins;b++) {
		if (bins[b]<=threshold) {  continue;  }
		// check if bin b shrunk below the minimum diameter
		if (radii[b] <= m_params.minBin/1e6) {
			newBins[0] += bins[b];
			continue;
		}
		Vnew = 3.14159265359*powf(radii[b],3.0) *4./3.;  // in cubic meters
		b1 = floor((radii[b]*2.0e6) / m_params.binSize);  // calculate new drop diameter bin
		if (b1>=m_params.numBins-1) { // we've got bigger drops than our bins can handle
			b1 = m_params.numBins-2;  b2 = b1+1;
			y1 = 0.0;
			y2 = bins[b];
		} else {
			b2 = b1+1;
			// Calculate the relative distribution between the two bins
			y2 = bins[b] * (Vnew-m_params.bVols[b1]) / (m_params.bVols[b2]-m_params.bVols[b1]);
			y1 = bins[b] - y2;
		}
		if (fabs(y1)<threshold) {
			y1 = 0.0;
			y2 = bins[b];
		} else if (fabs(y2)<threshold) {
			y2 = 0.0;
			y1 = bins[b];
		} else if (y1<0 || y2<0) {
			printf("y12: %e %e %e %e %e %d %d %d %e %e %e\n",y1,y2,bins[b1],bins[b2],radii[b],b,b1,b2,(Vnew-m_params.bVols[b1]) / (m_params.bVols[b2]-m_params.bVols[b1]),m_params.bDiams[b1]/2e6,m_params.bDiams[b2]/2e6);
		}
		newBins[b1] += y1;
		newBins[b2] += y2;
	}
	for (int b=0;b<m_params.numBins;b++) {
		bins[b] = newBins[b];
	}
}

void ParticleSystem::chemistry_serial(
		      float *oldPos,               // output: new velocity
              float *oldScalar,               // input: sorted positions
              float *oldDiamBins,               // input: sorted velocities
              //float *colors,
              int    numParticles)
{

    float delta_temp = 0.0;
    float delta_press = 0.0;

    float y1,y2;
    float Vnew;
    int b1,b2;

    float newBins[NUM_DIAM_BINS];

    // get address in grid
    float scalar;
    float particle_mass;

    float d10;
for (int index=0;index<numParticles;index++) {
	//printf("%d %d\n",index,numParticles);
    scalar = oldScalar[index];
    //float4 pos = FETCH(oldPos, index);       // macro does either global read or texture fetch
    particle_mass = oldPos[index*4+3];
    for (int b=0;b<m_params.numBins;b++) {
    	newBins[b] = oldDiamBins[index*m_params.numBins+b];
    }

    	RK4(scalar,newBins,m_params.dt,particle_mass);

    d10 = 0.0;
	for (int b=0;b<m_params.numBins;b++) {
		//TEMPORARY LINE:
		//newBins[b] = 0.0;
		oldDiamBins[index*m_params.numBins+b] = newBins[b];
		d10 += newBins[b]*m_params.bDiams[b];
	}

	if (index==1) {
		printf("d10 %f %f %f %f %f\n",d10,newBins[0],newBins[1],newBins[18],newBins[19]);
	}

	for (int b=0;b<m_params.numBins;b++) {
		newBins[b] = 0.0;
	}
 }
}

// step the simulation
void ParticleSystem::update(float deltaTime) {
	float* dPos;
	int i,k;

 	int i1, i2; //, ii;
	int j1, j2; //, jj;
	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;
	int gindex;
	float w1, w2, w3, w4;
	float denom;

	n_iter++;
	//printf("vols %e %e\n",cellVolume(make_uint3(0,0,0)),cellVolume(make_uint3(1,0,0)));
	//cin.ignore();

	/*
	for (int i=0;i<m_params.numCells.x;i++) {
		for (int k=0;k<m_params.numCells.z;k++) {
			for (int j=0;j<4;j++) {
				m_UCorrect[i*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.z+j] = m_UCorrect[i*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.z+j+1];
			}
			//m_UCorrect[i*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.z+4] = calculated correction;
		}
	}
	m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2))
	(m_Z[k+1]-m_Z[k])*m_darc*m_R[m_numVelCells.x]
	mat UCorr = zeros<mat>(m_params.numCells.x+m_params.numCells.z-2,m_params.numCells.x+m_params.numCells.z-2);
	for (int i=0;i<m_params.numCells.x-1;i++) {
		for (int k=0;k<m_params.numCells.z-1;k++) {
			// flow through the top of the cell
			UCorr[i][k] = m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2));
*/
	if (m_bUseOpenGL) {
		printf("in this dpos definition\n");
		dPos = (float*) (mapGLBufferObject(&m_cuda_posvbo_resource));
	} else {
		dPos = (float*) (m_cudaPosVBO);
	}
	// update constants
	printf("Setting parameters\n");
	setParameters(&m_params);
	//setCellParameters(&m_cellparams);
	printf("Parameters set\n");
	uint counter;
	//copyArrayFromDevice(m_hPos, dPos, 0, sizeof(float)*m_maxNumParticles*4);
	//printf("before int: %f %f %f %f\n",m_hPos[0],m_hPos[1],m_hPos[2],m_hPos[3]);
	// integrate
	threadSync();
/* DIAGNOSTIC
	copyArrayToHost(m_hSortedPos,      dPos,         0, sizeof(float) * m_maxNumParticles * 4);
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d VVVSSS. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		} else {
			if (m_hSortedPos[i*4]>0.2) {
				printf("This is the earliest SIGN\n");
			}
		}
	}
*/
//	copyArrayToDevice(m_dVel, m_hVel,  0, sizeof(float) * m_maxNumParticles * 4);
//	copyArrayToDevice(m_dScalar, m_hScalar,  0, sizeof(float) * m_maxNumParticles);
//	copyArrayToDevice(m_dDiamBins, m_hDiamBins,  0, sizeof(float) * m_maxNumParticles * m_params.numBins);

	printf("Integrating system\n");
	integrateSystem(dPos,
					m_dVel, //m_dScalar
					deltaTime,
					//m_numParticles);
					m_maxNumParticles);
	threadSync;
	printf("System integrated\n");
	copyArrayToHost(m_hSortedPos,      dPos,         0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToHost(m_hVel,      m_dVel,         0, sizeof(float) * m_maxNumParticles * 4);
	threadSync();
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d VVVSSS. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		} else {
			if (m_hSortedPos[i*4]>0.2) {
				printf("This is an EVEN Earlier SIGN %d %f %f %f %e\n",i,m_hSortedPos[i*4],m_hSortedPos[i*4+1],m_hSortedPos[i*4+2],m_hSortedPos[i*4+3]);
				//printf("Subsequent:                  %d %f %f %f %e\n",i,m_hSortedPos[(i+1)*4],m_hSortedPos[(i+1)*4+1],m_hSortedPos[(i+1)*4+2],m_hSortedPos[(i+1)*4+3]);
				printf("Vels:                        %d %f %f %f %e\n",i,m_hVel[i*4],m_hVel[i*4+1],m_hVel[i*4+2],m_hVel[i*4+3]);
			}
		}
	}
	// -----------------------
	//   PREPPING FOR FLUXES
	// -----------------------

	calcNewHashMax(m_dGridParticleHash, // determine this based on the cell the particle is in
			m_dGridParticleIndex, // determine index based on the thread location
			dPos, // use position to calc the hash
			m_maxNumParticles);
	printf("Hash calculated\n");
	// sort particles based on hash
	sortParticles(m_dGridParticleHash, // array by which to sort
			m_dGridParticleIndex, // array gets rearraged based on hash sorting
			m_maxNumParticles);
	printf("Particles sorted\n");
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	// first declared in particleSystem.cuh
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d VVVSSS! %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		} else {
			if (m_hSortedPos[i*4]>0.2) {
				printf("This is a Sanity Check\n");
			}
		}
	}

	printf("Reorder data\n");
	reorderDataAndFindCellStart(m_dCellStart,
								m_dCellEnd,
								m_dSortedPos,
								m_dSortedVel,
								m_dSortedScalar,
								m_dSortedDiamBins,
								m_dGridParticleHash,
								m_dGridParticleIndex,
								dPos,
								m_dVel,
								m_dScalar,
								m_dDiamBins,
								m_maxNumParticles,
								m_numParticles,
								m_numTotalCells);
//OLDER CODE DEBUG:
//								m_numTotalVelCells);
	printf("Data reordered %d %d %d\n",m_maxNumParticles,m_numParticles,m_numTotalVelNodes);

	copyArrayToHost(m_hSortedPos,      m_dSortedPos,         0, sizeof(float) * m_maxNumParticles * 4);

/* DIAGNOSTIC */
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0.0) {
			printf("%d vs. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		} else {
			if (m_hSortedPos[i*4]>0.2) {
				printf("This is an Earlier SIGN\n");
			}
		}
	}

	copyArrayToHost(m_hSortedScalar,   m_dSortedScalar,      0, sizeof(float) * m_maxNumParticles);
	copyArrayToHost(m_hSortedDiamBins, m_dSortedDiamBins,    0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	copyArrayToHost(m_hParticleIndex,  m_dGridParticleIndex, 0, sizeof(uint) * m_maxNumParticles);
	copyArrayToHost(m_hParticleHash,   m_dGridParticleHash,  0, sizeof(uint) * m_maxNumParticles);
	copyArrayToHost(m_hCellStart,   m_dCellStart,      0, sizeof(uint) * m_numTotalCells);
	copyArrayToHost(m_hCellEnd,     m_dCellEnd,        0, sizeof(uint) * m_numTotalCells);
	printf("Process boundaries\n");
/*
	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {
			printf("i%d ",(int)m_hVelCellStart[k*m_numVelCells.x+i]);
		}
		printf("\n");
	}
	//cin.ignore();
	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {
			printf("i%d ",(int)m_hVelCellEnd[k*m_numVelCells.x+i]);
		}
		printf("\n");
	}
*/

	//cin.ignore();

	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}
	//cin.ignore();


	boundaryFluxHybrid(
					m_hSortedPos, // send off the sorted positions
					m_hSortedScalar, // send off the sorted velocities
					m_hSortedDiamBins,
					m_hParticleHash,
					m_hParticleIndex,
					m_hCellStart,
					m_hCellEnd,
					m_maxNumParticles,
					m_numParticles,
					m_dens);
					//counter);

	printf("num of particles 3: %d\n",m_numParticles);

/*
	if (n_iter>m_histLength) {
		for (int i=0;i<m_params.numVelCells.x;i++) {
			for (int k=0;k<m_params.numVelCells.z;k++) {
				if (i==0) {
					m_params.dUr[i][k] = 1.0 * (m_dens[k*m_params.numVelCells.x+i]-m_dens[k*m_params.numVelCells.x+i+1]);
				} else if (i == m_params.numVelCells.x-1) {
					m_params.dUr[i][k] = 1.0 * (m_dens[k*m_params.numVelCells.x+i-1]-m_dens[k*m_params.numVelCells.x+i]);
				} else {
					m_params.dUr[i][k] = 1.0 * (m_dens[k*m_params.numVelCells.x+i-1]-m_dens[k*m_params.numVelCells.x+i+1]);
				}
				if (k==0) {
					m_params.dUz[i][k] = 1.0 * (m_dens[k*m_params.numVelCells.x+i]-m_dens[(k+1)*m_params.numVelCells.x+i]);
				} else if (k == m_params.numVelCells.z-1) {
					m_params.dUz[i][k] = 1.0 * (m_dens[(k-1)*m_params.numVelCells.x+i]-m_dens[k*m_params.numVelCells.x+i]);
				} else {
					m_params.dUz[i][k] = 1.0 * (m_dens[(k-1)*m_params.numVelCells.x+i]-m_dens[(k+1)*m_params.numVelCells.x+i]);
				}
			}
		}
	}
*/
	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well2, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}
	//cin.ignore();

	//m_numParticles,
	//m_numTotalVelCells);

	//copyArrayToDevice(dPos,        m_hPos,      0, sizeof(float) * m_maxNumParticles * 4);
	//copyArrayToDevice(m_dScalar,   m_hScalar,   0, sizeof(float) * m_maxNumParticles);
	//copyArrayToDevice(m_dDiamBins, m_hDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);

	copyArrayToDevice(dPos,        m_hSortedPos,      0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToDevice(m_dScalar,   m_hSortedScalar,   0, sizeof(float) * m_maxNumParticles);
	copyArrayToDevice(m_dDiamBins, m_hSortedDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);

	// ------------------------
	//   PREPPING FOR BALANCE
	// ------------------------
	calcNewHashMax(m_dGridParticleHash, // determine this based on the cell the particle is in
				   m_dGridParticleIndex, // determine index based on the thread location
				   dPos, // use position to calc the hash
				   m_maxNumParticles);
	printf("Hash calculated\n");

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, // array by which to sort
				  m_dGridParticleIndex, // array gets rearraged based on hash sorting
				  m_maxNumParticles);
	printf("Particles sorted\n");
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	// first declared in particleSystem.cuh
	printf("Reorder data\n");
	reorderDataAndFindCellStart(m_dCellStart,
								m_dCellEnd,
								m_dSortedPos,
								m_dSortedVel,
								m_dSortedScalar,
								m_dSortedDiamBins,
								m_dGridParticleHash,
								m_dGridParticleIndex,
								dPos,
								m_dVel,
								m_dScalar,
								m_dDiamBins,
								m_maxNumParticles,
								m_numParticles,
								m_numTotalCells);
	printf("Data reordered\n");
/*
	copyArrayToHost(m_hSortedPos,      m_dSortedPos,         0, sizeof(float) * m_maxNumParticles * 4);
	printf("Boundaries processed %d %d\n",m_numTotalVelCells,m_numTotalCells);
	//m_numParticles = counter;
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d Vs. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		} else {
			if (m_hSortedPos[i*4]>0.2) {
				printf("This is a SIGN\n");
			}
		}
	}
*/
	bool balanceCuda = true;
	printf("Balance cells\n");
	//counter = m_numParticles;

	copyArrayToHost(m_hSortedPos,     m_dSortedPos,         0, sizeof(float)*m_maxNumParticles*4);
	 copyArrayToHost(m_hSortedScalar,  m_dSortedScalar,      0, sizeof(float)*m_maxNumParticles);
	 copyArrayToHost(m_hSortedDiamBins,m_dSortedDiamBins,    0, sizeof(float)*m_maxNumParticles * m_params.numBins);
	 copyArrayToHost(m_hParticleIndex, m_dGridParticleIndex, 0, sizeof(uint)*m_maxNumParticles);
	 copyArrayToHost(m_hParticleHash,  m_dGridParticleHash,  0, sizeof(uint)*m_maxNumParticles);
	 copyArrayToHost(m_hCellStart,     m_dCellStart, 0, sizeof(uint)*m_numTotalCells);
	 copyArrayToHost(m_hCellEnd,       m_dCellEnd, 0, sizeof(uint)*m_numTotalCells);

/*
	for (int k=0;k<m_numCells.z;k++) {
		for (int i=0;i<m_numCells.x;i++) {
			printf("i%d ",m_hCellStart[k*m_numCells.x+i]);
		}
		printf("\n");
	}
	printf("total cells1: %d %d %d\n",m_numTotalCells,m_numCells.z,m_numCells.x);
	//cin.ignore();
	for (int k=0;k<m_numCells.z;k++) {
		for (int i=0;i<m_numCells.x;i++) {
			printf("i%d ",(int)m_hCellEnd[k*m_numCells.x+i]);
		}
		printf("\n");
	}
	//cin.ignore();
*/
	printf("num of particles 2: %d\n",m_numParticles);

	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0.0) {
			printf("%d veeS. %d     %f %e %e\n",i,m_numParticles,m_hSortedPos[i*4+3],m_hSortedPos[(i-1)*4+3],m_hSortedPos[0+3]);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		}
	}
	printf("Passed veeS\n");

	if (!balanceCuda) {
/*
	balanceCells(
	 // m_hPos,
	 // m_hScalar,
	 m_hSortedPos,         // send off the sorted positions
	 m_hSortedScalar,         // send off the sorted scalars
	 m_hSortedDiamBins,
	 m_hParticleHash,
	 m_hParticleIndex,
	 m_hCellStart,
	 m_hCellEnd,
	 m_maxNumParticles,
//	 counter,
	 m_numParticles);
//	 m_numTotalCells);
*/
	 //copyArrayToDevice(dPos,      m_hPos,    0, sizeof(float)*m_maxNumParticles*4);
	 //copyArrayToDevice(m_dScalar, m_hScalar, 0, sizeof(float)*m_maxNumParticles);
	 copyArrayToDevice(m_dSortedPos,      m_hSortedPos,    0, sizeof(float)*m_maxNumParticles*4);
	 copyArrayToDevice(m_dScalar, m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles);
	 copyArrayToDevice(m_dSortedDiamBins, m_hSortedDiamBins, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);

	} else {

	balanceCellsCuda(
			// m_dPos,
			// m_dScalar,
			m_dSortedPos, // send off the sorted positions
			m_dSortedScalar, // send off the sorted scalars
			m_dSortedDiamBins,
			m_dGridParticleHash,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_maxNumParticles,
			m_numParticles,
			//counter,
			m_numTotalCells);
	threadSync();

	//copyArrayToHost  (m_hSortedDiamBins,     m_dSortedDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well3, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}
	}

/*
	copyArrayToHost  (m_hSortedPos,    m_dSortedPos,      0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToHost  (m_hSortedScalar, m_dSortedScalar,   0, sizeof(float) * m_maxNumParticles);
	copyArrayToHost  (m_hDiamBins,     m_dSortedDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	copyArrayToDevice(dPos,            m_hSortedPos,      0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToDevice(m_dScalar,       m_hSortedScalar,   0, sizeof(float) * m_maxNumParticles);
	copyArrayToDevice(m_dDiamBins,     m_hDiamBins,       0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	//copyArrayToDevice(m_dDiamBins, m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);
*/
	//m_numParticles = counter;
	printf("Cells balanced, np %d\n", m_numParticles);
/*
	copyArrayToHost(m_hSortedPos,     m_dSortedPos,         0, sizeof(float) * m_maxNumParticles * 4);

	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0.0) {
			printf("%d V. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		}
	}
*/
	//int prevNumParticles = m_numParticles;
	calcNewHashMax( m_dGridParticleHash, // determine this based on the cell the particle is in
					m_dGridParticleIndex, // determine index based on the thread location
					m_dSortedPos, // use position to calc the hash
					m_maxNumParticles);
	printf("Hash calculated\n");
	// sort particles based on hash
	sortParticles(m_dGridParticleHash, // array by which to sort
				  m_dGridParticleIndex, // array gets rearranged based on hash sorting
				  m_maxNumParticles);
	printf("Particles sorted 3\n");
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	// first declared in particleSystem.cuh
	printf("Reorder data\n");

	reorderDataAndFindCellStart(m_dCellStart,
								m_dCellEnd,
								dPos,
								m_dVel,
								m_dScalar,
								m_dDiamBins,
								m_dGridParticleHash,
								m_dGridParticleIndex,
								m_dSortedPos,
								m_dSortedVel,
								m_dSortedScalar,
								m_dSortedDiamBins,
								m_maxNumParticles,
								m_numParticles,
								m_numTotalCells);
	copyArrayToHost(m_hSortedPos,     dPos,         0, sizeof(float) * m_maxNumParticles * 4);
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d S. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		}
	}

	printf("Data reordered\n");
	// process collisions
	printf("Process chemistry\n");

	if (true) {
		chemistry(dPos,
				  m_dScalar,
				  m_dDiamBins,
				  m_dColors,
				  m_maxNumParticles,
				  m_numParticles);
	} else if (true) {
		copyArrayToHost(m_hSortedPos,      dPos,         0, sizeof(float) * m_maxNumParticles * 4);
		copyArrayToHost(m_hSortedScalar,   m_dScalar,      0, sizeof(float) * m_maxNumParticles);
		copyArrayToHost(m_hSortedDiamBins, m_dDiamBins,    0, sizeof(float) * m_maxNumParticles * m_params.numBins);
		chemistry_serial(m_hSortedPos,
				  m_hSortedScalar,
				  m_hSortedDiamBins,
				  m_numParticles);
		threadSync();
	}
	//cin.ignore();

	printf("Chemistry processed\n");

	// -----------------------
	//   PREPPING FOR MIXING
	// -----------------------
//	copyArrayToHost(m_hSortedPos,      m_dSortedPos,         0, sizeof(float) * m_maxNumParticles * 4);
//	copyArrayToHost(m_hSortedScalar,   m_dSortedScalar,      0, sizeof(float) * m_maxNumParticles);
//	copyArrayToHost(m_hSortedDiamBins, m_dSortedDiamBins,    0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	copyArrayToHost(m_hSortedPos,      dPos,         0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToHost(m_hSortedScalar,   m_dScalar,      0, sizeof(float) * m_maxNumParticles);
	copyArrayToHost(m_hSortedDiamBins, m_dDiamBins,    0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	copyArrayToHost(m_hCellStart,   m_dCellStart,      0, sizeof(uint) * m_numTotalCells);
	copyArrayToHost(m_hCellEnd,     m_dCellEnd,        0, sizeof(uint) * m_numTotalCells);
	printf("Copied to Host\n");

	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0) {
			printf("%d VS. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		}
	}

	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well3.5, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}

	printf("Off to mixing\n");
	if (true) {
		mixCurl(m_hSortedPos,
				m_hSortedScalar,
				m_hSortedDiamBins,
				m_hCellStart,
				m_hCellEnd,
				m_dens);
	}

	threadSync();

	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well4, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}

	copyArrayToHost  (m_hColors,        m_dColors,         0, sizeof(float) * m_maxNumParticles);
//	copyArrayToHost  (m_hSortedPos,     m_dSortedPos,      0, sizeof(float) * m_maxNumParticles * 4);
//	copyArrayToHost  (m_hVel,           m_dSortedVel,      0, sizeof(float) * m_maxNumParticles * 4);
//	copyArrayToHost  (m_hSortedDiamBins,m_dSortedDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);
//	copyArrayToHost  (m_hSortedScalar,  m_dSortedScalar,   0, sizeof(float) * m_maxNumParticles);
	printf("More host copying\n");
/*
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hVel[i*4]>3.0) {
			printf("This is a VEL sign:     %d %f %f %f %e\n",i,m_hVel[i*4],m_hVel[i*4+1],m_hVel[i*4+2],m_hVel[i*4+3]);
		}
	}
*/
	//cin.ignore();

	copyArrayToDevice(dPos,             m_hSortedPos,      0, sizeof(float) * m_maxNumParticles * 4);
//	copyArrayToDevice(m_dVel,           m_hVel,            0, sizeof(float) * m_maxNumParticles * 4);
	copyArrayToDevice(m_dDiamBins,      m_hSortedDiamBins, 0, sizeof(float) * m_maxNumParticles * m_params.numBins);
	copyArrayToDevice(m_dScalar,        m_hSortedScalar,   0, sizeof(float) * m_maxNumParticles);
	printf("Copied to device\n");

/*
	// -----------------------
	//   PREPPING FOR ADVECT
	// -----------------------
	//int prevNumParticles = m_numParticles;
	calcNewVelHashMax(m_dGridParticleHash, // determine this based on the cell the particle is in
					  m_dGridParticleIndex, // determine index based on the thread location
					  m_dSortedPos, // use position to calc the hash
					  m_maxNumParticles);
	printf("Hash calculated\n");
	// sort particles based on hash
	sortParticles(m_dGridParticleHash, // array by which to sort
				  m_dGridParticleIndex, // array gets rearraged based on hash sorting
				  m_maxNumParticles);
	printf("Particles sorted 2\n");
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	// first declared in particleSystem.cuh
	printf("Reorder data\n");
	reorderDataAndFindCellStart(m_dVelCellStart,
								m_dVelCellEnd,
								dPos,
								m_dVel,
								m_dScalar,
								m_dDiamBins,
								m_dGridParticleHash,
								m_dGridParticleIndex,
								m_dSortedPos,
								m_dSortedVel,
								m_dSortedScalar,
								m_dSortedDiamBins,
								m_maxNumParticles,
								m_numParticles,
								m_numTotalVelNodes);
	printf("num of particles 1: %d\n",m_numParticles);

	copyArrayToHost(m_hSortedPos,     dPos,         0, sizeof(float) * m_maxNumParticles * 4);
	for (int i=0;i<m_maxNumParticles;i++) {
		if (m_hSortedPos[i*4+3]<0.0) {
			printf("%d vS. %d\n",i,m_numParticles);
			if (i!=m_numParticles) {
				cin.ignore();
			}
			break;
		}
	}
	//cin.ignore();
*/
	printf("Data reordered\n");
	printf("Process advection\n");
	for (int i = 0; i < m_maxNumParticles * 3; i++) {
		m_hRandom[i] = frand();
	}
	copyArrayToDevice(m_dRandom, m_hRandom, 0, sizeof(float) * m_maxNumParticles * 3);

	//cin.ignore();

	advect(	m_dVel, // receive new velocities here
			dPos, // send off the sorted positions
			m_dGridParticleIndex, // contains old order of indices
			m_dVelCellStart,
			m_dVelCellEnd,
			m_maxNumParticles,
			m_numTotalVelNodes,
			m_dRandom,
			m_dUz,
			m_dUr,
			m_dEpsTKE,
			m_dNut,
			m_dGradZNut,
			m_dGradRNut,
			m_dDUz,
			m_dDUr
			);
	threadSync();
	printf("Advection processed\n");

	copyArrayToHost  (m_hVel,        m_dVel,         0, sizeof(float) * m_maxNumParticles*4);

	//for (int i=0;i<m_maxNumParticles;i++) {
	//	if (m_hSortedDiamBins[i*m_params.numBins]<1.0) {
	//		printf("well5, %f %d\n",m_hSortedDiamBins[i*m_params.numBins],i);
	//	}
	//}

	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
				m_cellParts[i*m_numCells.z+k] = 0;
		}
	}
	printf("Scrolled through cell parts\n");

	for (int j=0;j<m_numParticles;j++) {
		//if (m_hSortedDiamBins[j*m_params.numBins]<1.0) {
		//	printf("cellinfo: %f, %f (%d %d) %f %f %e %d\n",cellinfo,m_hSortedDiamBins[j*m_params.numBins],i,k,m_hSortedPos[j*4+0],m_hSortedPos[j*4+2],m_hSortedPos[j*4+3],m_loopCounter);
		//}
		i = (int)(m_hSortedPos[j*4]/m_params.cellSize.x);
		k = (int)(m_hSortedPos[j*4+2]/m_params.cellSize.z);
		gindex = calcGridHashHost(i,0,k);
		//printf("gindex taken %d/%d %d %d %d %f %e\n",j,m_numParticles,gindex,i,k,m_hSortedPos[j*4],m_hSortedPos[j*4+3]);
		m_cellScalar[gindex] += m_hSortedScalar[j] * m_hSortedPos[j*4+3];
		m_cellMass  [gindex] += m_hSortedPos[j*4+3];
		if (m_hSortedPos[j*4+3]<0.0) {
			printf("%d %f\n",j,m_hSortedPos[j*4+3]);
			cin.ignore();
		}
		//printf("checked sorted pos\n");
		m_cellParts [gindex] += 1;
		float cellinfo=0.0;
		for (int b=0;b<m_params.numBins;b++) {
			m_cellD10[gindex] += m_hSortedDiamBins[j*m_params.numBins+b]*b*m_params.binSize; //* m_hSortedPos[j*4+3];
			//m_cellDiams[k*m_numCells.x*m_numBins+i*m_numBins+b] += m_hSortedDiamBins[j*m_params.numBins+b] * m_hSortedPos[j*4+3];
			m_cellDiams[gindex*m_numBins+b] += m_hSortedDiamBins[j*m_params.numBins+b]; //* m_hSortedPos[j*4+3];
			cellinfo += m_hSortedDiamBins[j*m_params.numBins+b];
		}
		//printf("scrolled through bins\n");
		//if (m_hSortedDiamBins[j*m_params.numBins]<1.0) {
		//	printf("cellinfo: %f, %f (%d %d) %f %f %e %d\n",cellinfo,m_hSortedDiamBins[j*m_params.numBins],i,k,m_hSortedPos[j*4+0],m_hSortedPos[j*4+2],m_hSortedPos[j*4+3],m_loopCounter);
		//}
		m_cellRMSScalar[gindex] += powf(m_hSortedScalar[j]-m_cellAvgScalar[gindex],2.0)*m_hSortedPos[j*4+3];
	}

	// Velocity correction calculations
	printf("Begin velocity correction calcs\n");
	for (int i=0;i<m_params.numCells.x;i++) {
		for (int k=0;k<m_params.numCells.z;k++) {
			gindex = calcGridHashHost(i,0,k);
			for (int j=0;j<m_histLength-1;j++) {
				m_densHist[j*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i] = m_densHist[(j+1)*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i];
			}
			if (i==m_params.numCells.x-1 && k==m_params.numCells.z-1) {
				printf("Further upstream: %f\n",m_dens[gindex]);
			}
			m_densHist[(m_histLength-1)*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i] = m_dens[gindex];
		}
	}
	float dens_diff;
	if (n_iter>m_histLength) {
		for (int i=0;i<m_params.numCells.x;i++) {
			for (int k=0;k<m_params.numCells.z;k++) {
				gindex = calcGridHashHost(i,0,k);
				//m_dens[gindex] = 0.0;
				dens_diff = 0.0;
				for (int j=0;j<m_histLength;j++) {
					//m_dens[gindex] += m_densHist[j*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i]-m_hRhoCells[i*m_params.numCells.z+k];
					dens_diff += m_densHist[j*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i]-m_hRhoCells[i*m_params.numCells.z+k];
					//if (i==m_params.numCells.x-1 && k==m_params.numCells.z-1) {
					//	printf("building %f %f\n",m_dens[gindex],m_densHist[j*m_params.numCells.x*m_params.numCells.z+k*m_params.numCells.x+i]);
					//}
				}
				//m_dens[gindex] = m_dens[gindex]/m_histLength;
				//printf("This should be close to 1: %f\n",m_params.dUr[i][k]);
				//m_params.dUr[i][k] = m_dens[gindex]/m_histLength;
				m_hDUr[i*m_params.numCells.z+k] = dens_diff/m_histLength;
				m_hDUz[i*m_params.numCells.z+k] = m_hDUr[i*m_params.numCells.z+k];
				//RMKrho
				//m_params.dUr[i][k] = m_dens[gindex]/m_histLength - m_cellRho[i][k];
				//m_params.dUz[i][k] = m_params.dUr[i][k];
			}
		}
	}

	copyArrayToDevice(m_dDUr, m_hDUr, 0, sizeof(float) * m_numTotalCells);
	copyArrayToDevice(m_dDUz, m_hDUz, 0, sizeof(float) * m_numTotalCells);

	//    copyArrayFromDevice(m_hPos, m_dSortedPos, 0, sizeof(float)*m_maxNumParticles*4);
	printf("num particles: %d\n", m_numParticles);
	//cin.ignore();
	printf("reemerge\n");

	// this is only for coloring purposes:
	//copyArrayToHost(m_hVel, m_dVel, 0,sizeof(float) * m_maxNumParticles * 4);

	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	//printf("Check 1\n");
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;
	//printf("Check 3 %f\n",m_hScalar[0]);
	//printf("color: %f\n", m_hColors[0]);
	//for (int i = 0; i < m_maxNumParticles; i++) {
	float binscalar;
	for (int i = 0; i < m_numParticles; i++) {
		//binscalar = 0.0;
		//for (int b=0;b<m_params.numBins;b++) {
		//	binscalar += m_hSortedDiamBins[i*m_params.numBins+b];
		//}
		//colorRamp(binscalar/1.2, ptr);

		//colorRamp(m_hSortedPos[i*4+3]*2000000, ptr);
		//colorRamp(m_hVel[i*4+2]/5., ptr);

		//colorRamp(m_hColors[i], ptr);  // for super-saturation (change color scaling, too)
		colorRamp(m_hSortedScalar[i], ptr);
		ptr += 3;
		*ptr++ = 1.0f;
	}
	printf("Check 4\n");
	glUnmapBufferARB(GL_ARRAY_BUFFER);
	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	if (m_bUseOpenGL) {
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}
	printf("reemerge2\n");

	if (m_loopCounter%4==2) {
		glBegin(GL_LINES);
		glColor3f(0.0, 0.6, 0.0);
		printf("do these lines!\n");
		for (int i = 0; i <= m_params.numCells.z; i++) {
			glVertex3f(m_params.worldOrigin.x, 0.0, m_params.cellSize.z * i);
			glVertex3f(m_params.worldOrigin.x+m_params.worldSize.x, 0.0, m_params.cellSize.z * i);
		}

		for (int i = 0; i <= m_params.numCells.x; i++) {
			glVertex3f(m_params.cellSize.x * i, 0.0, m_params.worldOrigin.z);
			glVertex3f(m_params.cellSize.x * i, 0.0, m_params.worldOrigin.z+m_params.worldSize.z);
		}
		glEnd();
	}
	if (m_loopCounter%4==0) {
		glBegin(GL_LINES);
		glColor3f(0.0, 0.6, 0.6);
		for (int i = 0; i < m_params.numVelNodes.z; i++) {
			glVertex3f(m_R[0], 0.0, m_Z[i]);
			glVertex3f(m_R[m_params.numVelNodes.x-1], 0.0, m_Z[i]);
		}
		for (int i = 0; i < m_params.numVelNodes.x; i++) {
			glVertex3f(m_R[i], 0.0, m_Z[0]);
			glVertex3f(m_R[i], 0.0, m_Z[m_params.numVelNodes.z-1]);
		}
		glEnd();
	}

	m_loopCounter++;
	m_loopCounterTotal++;
	printf("Loop Counter: %d\n",m_loopCounterTotal);

	if (m_loopCounterTotal==4001) {
	//if (m_loopCounterTotal==8000) {
		saveCells();
		printf("Saved!!!\n");
		cin.ignore();
	}
	if (m_loopCounterTotal==2000) {
	//if (m_loopCounterTotal==4000) {
        zeroCells();
	}
	if (m_loopCounterTotal==3000) {
	//if (m_loopCounterTotal==6000) {
        setAvgCells();
	}
	if (m_loopCounterTotal==4001) {
	//if (m_loopCounterTotal==4000) {
	//if (m_loopCounterTotal==8000) {
		exit(EXIT_SUCCESS);
	}
	//cin.ignore();
}

void ParticleSystem::saveCells()
{
	FILE *fmass;
	//FILE *fdensity;
	FILE *fvolume;
	FILE *fscalar;
	FILE *fd10;
	FILE *fparts;
	FILE *frmsscalar;
	FILE *fdiam;
	FILE *fvelr;
	FILE *fvelz;
	uint3 index3;
	index3.y = 0;
	int gindex;
	//filename = directory+Uzfile;
	//fp = fopen(filename.c_str(),"r");
	// DomainSize
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot/";
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big/";
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big/";-nothot-big-refine
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-nothot-big-refine/";
	string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine-lighter/";
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet62-big-refine-lighter/";
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet85-big-refine/";
	//string savedir = "/home/rkeedy/Dropbox/CFD/BuoyantStrumJet63-big-refine/";
	string savefile;
	savefile = savedir+"massdata.txt";
	fmass    = fopen(savefile.c_str(),"w");
	//savefile = savedir+"densitydata.txt";
	//fdensity = fopen(savefile.c_str(),"w");
	savefile = savedir+"volumedata.txt";
	fvolume  = fopen(savefile.c_str(),"w");
	savefile = savedir+"scalardata.txt";
	fscalar  = fopen(savefile.c_str(),"w");
	savefile = savedir+"d10data.txt";
	fd10     = fopen(savefile.c_str(),"w");
	savefile = savedir+"partsdata.txt";
	fparts   = fopen(savefile.c_str(),"w");
	savefile = savedir+"rmsscalardata.txt";
	frmsscalar = fopen(savefile.c_str(),"w");
	savefile = savedir+"diamdata.txt";
	fdiam    = fopen(savefile.c_str(),"w");
	savefile = savedir+"velrdata.txt";
	fvelr    = fopen(savefile.c_str(),"w");
	savefile = savedir+"velzdata.txt";
	fvelz    = fopen(savefile.c_str(),"w");
	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
			index3.x = i;
			index3.z = k;
			gindex = calcGridHashHost(i,0,k);
			fprintf(fmass,"%e %d %d\n",m_cellMass[gindex]/m_loopCounter,i,k);
			//fprintf(fdensity,"%e %d %d\n",m_cellMass[gindex]/cellVolume(index3)/(m_loopCounter),i,k);
			fprintf(fvolume,"%e %d %d\n",cellVolume(index3),i,k);
			fprintf(fscalar,"%f %d %d\n",m_cellScalar[gindex]/m_cellMass[gindex],i,k);
			//fprintf(fd10,"%e %d %d\n",m_cellD10[gindex]/m_cellMass[gindex],i,k);
			fprintf(fd10,"%e %d %d\n",m_cellD10[gindex]/m_loopCounter,i,k);
			fprintf(fparts,"%d %d %d\n",m_cellParts[gindex],i,k);
			fprintf(frmsscalar,"%f %d %d\n",powf(m_cellRMSScalar[gindex]/m_cellMass[gindex],0.5),i,k);
			for (int b=0;b<m_numBins-1;b++) {
				//fprintf(fdiam,"%f ",m_cellDiams[k*m_numCells.x*m_numBins+i*m_numBins+b]/m_cellMass[gindex]);
				//fprintf(fdiam,"%e ",m_cellDiams[gindex*m_numBins+b]/m_cellMass[gindex]);
				fprintf(fdiam,"%e ",m_cellDiams[gindex*m_numBins+b]/m_loopCounter);
				//fprintf(fdiam,"%f ",m_cellDiams[i*m_numCells.z*m_numBins+k*m_numBins+b]);
			}
			//fprintf(fdiam,"%f\n",m_cellDiams[k*m_numCells.x*m_numBins+(i+1)*m_numBins-1]/m_cellMass[gindex]);
			//fprintf(fdiam,"%e\n",m_cellDiams[(gindex+1)*m_numBins-1]/m_cellMass[gindex]);
			fprintf(fdiam,"%e\n",m_cellDiams[(gindex+1)*m_numBins-1]/m_loopCounter);
			//fprintf(fdiam,"%f\n",m_cellDiams[i*m_numCells.z*m_numBins+(k+1)*m_numBins-1]);
		}
	}
/*
	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {
			fprintf(fvelr,"%e %d %d\n",m_hUr[i*m_numVelCells.z+k]+m_hGradRNut[i*m_numVelCells.z+k]/m_params.schmidt/m_params.density_rho,i,k);
			fprintf(fvelz,"%e %d %d\n",m_hUz[i*m_numVelCells.z+k]+m_hGradZNut[i*m_numVelCells.z+k]/m_params.schmidt/m_params.density_rho,i,k);
		}
	}
	*/
	fclose(fmass);
	//fclose(fdensity);
	fclose(fvolume);
	fclose(fscalar);
	fclose(fd10);
	fclose(fparts);
	fclose(frmsscalar);
	fclose(fdiam);
	fclose(fvelr);
	fclose(fvelz);
	cin.ignore();
}

void ParticleSystem::zeroCells()
{
	int gindex;
	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
			gindex = calcGridHashHost(i,0,k);
			m_cellMass[gindex]=0.0;
			m_cellScalar[gindex]=0.0;
			m_cellD10[gindex]=0.0;
			m_cellParts[gindex]=0.0;
			m_cellAvgScalar[gindex]=0.0;
			for (int b=0;b<m_numBins;b++) {
				m_cellDiams[k*m_numCells.x*m_numBins+i*m_numBins+b]=0.0;
			}
		}
	}
	m_loopCounter = 0;
}

void ParticleSystem::setAvgCells()
{
	for (int i=0;i<m_numCells.x;i++) {
		for (int k=0;k<m_numCells.z;k++) {
			m_cellAvgScalar[i*m_numCells.z+k]=m_cellScalar[i*m_numCells.z+k]/m_cellMass[i*m_numCells.z+k];
			m_cellRMSScalar[i*m_numCells.z+k]=0.0;
		}
	}
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numTotalCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numTotalCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numTotalCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
	int local_num_bins = NUM_DIAM_BINS;
	//int local_num_bins = 10;
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);
    copyArrayFromDevice(m_hScalar, m_dScalar, 0, sizeof(float)*count);
    copyArrayFromDevice(m_hDiamBins, m_dDiamBins, 0, sizeof(float)*count*m_params.numBins);

    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            cuda_vbo_resource = m_cuda_posvbo_resource;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;
    }

    //copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_maxNumParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    //int local_num_bins = 20;
    int local_num_bins = NUM_DIAM_BINS;

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
            }
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;

        case SCALAR:
            copyArrayToDevice(m_dScalar, data, start*1*sizeof(float), count*1*sizeof(float));
            break;

        case BINS:
            copyArrayToDevice(m_dDiamBins, data, start*m_params.numBins*sizeof(float), count*m_params.numBins*sizeof(float));
            break;
    }
}

void
//ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint maxNumParticles, uint numParticles)
{
    srand(1973);
    uint3 velgridpos;

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                //if (i < MaxParticles)
                if (i < maxNumParticles)
                {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    if (i<numParticles) {
                    	//m_hPos[i*4+3] = cellVolume(calcGridPos(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]))*m_params.celldensity/m_params.avgParticlesPerCell;
                    	velgridpos = calcVelGridPos(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]);
                    	m_hPos[i*4+3] = cellVolume(calcGridPos(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]))* m_hRho[velgridpos.x*m_numVelNodes.z+velgridpos.z] /m_params.avgParticlesPerCell;
                    	//printf("initialize mass %e\n",m_hPos[i*4+3]);
                    } else {
                    	m_hPos[i*4+3] = -1.0f;
                    }

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;

                    m_hScalar[i] = 0.1f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
	//int local_num_bins = 20;
	int local_num_bins = NUM_DIAM_BINS;

    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
            	printf("Here goes the reset %f %f\n",m_params.worldSize.x,m_params.worldSize.z);
                int p = 0, v = 0, s = 0;

                //for (uint i=0; i < m_numParticles; i++)
                for (uint i=0; i < m_maxNumParticles; i++)
                {
                	//printf("Count up! %i\n",i);
                    float point[3];
                    point[0] = frand();
                    point[2] = frand();
                    //m_hPos[p++] = (m_R[m_gridSize.x]-m_R[0]) * point[0] + m_R[0];
                    m_hPos[p++] = m_params.worldSize.x * point[0] + m_params.worldOrigin.x;
                    m_hPos[p++] = 0.0;
                    //m_hPos[p++] = (m_Z[m_gridSize.z]-m_Z[0]) * point[2] + m_Z[0];
                    m_hPos[p++] = m_params.worldSize.z * point[2] + m_params.worldOrigin.z;
                    // RMK: check if valid particle or buffer
                    if (i<m_numParticles) {
                    	//m_hPos[p++] = 0.1f; // radius
                    	m_hPos[p++] = cellVolume(calcGridPos(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]))*interpRho(m_hPos[i*4],m_hPos[i*4+2])/m_params.avgParticlesPerCell;
                    } else {
                    	m_hPos[p++] = -1.0f;
                    }
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hScalar[s++] = 0.0f;

                    //printf("Print loop %d\n",i);

                    for (int b=1;b<m_params.numBins;b++) {
                    	m_hDiamBins[i*m_params.numBins+b] = 0.0;
                    }
                    m_hDiamBins[i*m_params.numBins] = 1.0;
                }
            }
            break;

        case CONFIG_UNIFORM:
        {
        	printf("Here goes the uniform reset %f %f\n",m_params.worldSize.x,m_params.worldSize.z);
            int p = 0, v = 0, s = 0, d = 0;

            //for (uint i=0; i < m_numParticles; i++)
            for (int i=0; i < m_params.numCells.x; i++) {
                for (int k=0; k < m_params.numCells.z; k++) {
                	for (int j=0; j < m_params.avgParticlesPerCell; j++) {
                		//printf("do pos   ");
						m_hPos[p++] = ((float)i+0.5)*m_params.cellSize.x;
						m_hPos[p++] = 0.0;
						m_hPos[p++] = ((float)k+0.5)*m_params.cellSize.z;
						m_hPos[p++] = cellVolume(make_uint3(i,0,k))*m_hRhoCells[i*m_params.numCells.z+k]/m_params.avgParticlesPerCell;


						// RMK: check if valid particle or buffer
						//printf("do scalars   ");
						m_hVel[v++] = 0.0f;
						m_hVel[v++] = 0.0f;
						m_hVel[v++] = 0.0f;
						m_hVel[v++] = 0.0f;
						m_hScalar[s++] = 0.0f;

						//printf("Print loop %d %d %d     ",i,k,j);

						for (int b=1;b<m_params.numBins;b++) {
							m_hDiamBins[d*m_params.numBins+b] = 0.0;
						}
						//printf("doing diams %d   ",d);
						m_hDiamBins[d*m_params.numBins] = m_params.conc_nuc_vol*cellVolume(make_uint3(i,0,k))/m_params.avgParticlesPerCell*1.0e6;  // 1e6 needed to convert concentration from cm^-3 to m^-3
						//printf("%d %d %f %e\n",i,k,m_hDiamBins[d*m_params.numBins],cellVolume(make_uint3(i,0,k)));
						d++;

						//printf("inced d %d\n",m_maxNumParticles);
                	}
                }
				//cin.ignore();
            }
            printf("done with those cells\n");
			for (int i=m_params.numCells.x*m_params.numCells.z*m_params.avgParticlesPerCell; i<m_maxNumParticles; i++) {
				m_hPos[p++] = 0.0;
				m_hPos[p++] = 0.0;
				m_hPos[p++] = 0.0;
				m_hPos[p++] = -1.0;
				// RMK: check if valid particle or buffer
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hScalar[s++] = 0.0f;

				//printf("Print loop %d\n",i);

				for (int b=0;b<m_params.numBins;b++) {
					m_hDiamBins[d*m_params.numBins+b] = 0.0;
				}
				//m_hDiamBins[d*m_params.numBins] = 1.0;
				d++;
			}
        }

    	break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                // RMK modification
                gridSize[1] = 0.0;
                initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_maxNumParticles, m_numParticles);
            }
            break;
    }

    printf("Setting arrays\n");
    setArray(POSITION, m_hPos, 0, m_maxNumParticles);
    setArray(VELOCITY, m_hVel, 0, m_maxNumParticles);
    setArray(SCALAR, m_hScalar, 0, m_maxNumParticles);
    setArray(BINS, m_hDiamBins, 0, m_maxNumParticles);
}
