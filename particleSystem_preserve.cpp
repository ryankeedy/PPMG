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

//ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float deltatime, bool bUseOpenGL) :
ParticleSystem::ParticleSystem(uint maxNumParticles, uint numParticles, uint3 numVelCells, uint3 numCells, float deltatime, bool bUseOpenGL) :
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
    m_numVelCells(numVelCells),
    m_timer(NULL),
    m_solverIterations(1),
    m_maxNumParticles(maxNumParticles)
{
	printf("In ParticleSystem\n");
	//char Zfile[] = "/home/user/Dropbox/CFD/TurbJetNuCoarserWide/ZVert.txt";
	//char Rfile[] = "/home/user/Dropbox/CFD/TurbJetNuCoarserWide/RVert.txt";
	char Zfile[] = "/home/rkeedy/Dropbox/CFD/TurbJetNuCoarserWide/ZVert.txt";
	char Rfile[] = "/home/rkeedy/Dropbox/CFD/TurbJetNuCoarserWide/RVert.txt";
	//char Zfile[] = "/home/keedy/Dropbox/CFD/TurbJetNuCoarserWide/ZVert.txt";
	//char Rfile[] = "/home/keedy/Dropbox/CFD/TurbJetNuCoarserWide/RVert.txt";

	m_numTotalCells = m_numCells.x*m_numCells.y*m_numCells.z;
	m_numTotalVelCells = m_numVelCells.x*m_numVelCells.y*m_numVelCells.z;
    //float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.numVelCells = m_numVelCells;
    m_params.numCells = m_numCells;
    m_params.numTotalVelCells = m_numTotalVelCells;
    m_params.numTotalCells = m_numTotalCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f / 10.0;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);
    //m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    //float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    //m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.celldensity = 1.0;
    m_params.dt = deltatime;

    m_params.darc = 1.0;
    m_params.schmidt = 0.7;
    m_params.density_rho = 1.0;
    m_params.maxParticlesPerCell = 10;
    m_params.minParticlesPerCell = 6;
    m_params.avgParticlesPerCell = (m_params.maxParticlesPerCell+m_params.minParticlesPerCell)/2;
    m_params.null_grid_value = m_numCells.x*m_numCells.y*m_numCells.z+1;
    m_params.null_velgrid_value = m_numVelCells.x*m_numVelCells.y*m_numVelCells.z+1;
    m_params.jet_radius = 0.01;

    printf("That was a bunch of initializing; now bins!\n");

    m_params.numBins = m_params.NUM_DIAM_BINS;
    m_params.minBin =  0.1;
    m_params.binSize = 0.5;
	for (int n=1;n<m_params.numBins;n++) {
		 m_params.bDiams[n] = n*m_params.binSize;
		 m_params.bVols[n] = 3.14159265359*powf((m_params.bDiams[n]/2.0e6),3.0) *4./3.; // # in cubic meters
	}
	printf("Out of loop\n");
	m_params.bDiams[0] = m_params.minBin;
	m_params.bVols[0] = 3.14159265359*powf((m_params.bDiams[0]/2.0e6),3.0) *4./3.; // # in cubic meters
	printf("All done with bins\n");

	m_params.lambda_h2o = 2257e3;
	m_params.conc_nuc_vol = m_params.NUCLEII_PER_M3;
	m_params.density_h2o = 1000.0;
	m_params.density_air = 1.23;
	m_params.Cp = 1.012 * 1000;
	m_params.patm = 101.0e3;
	m_params.gamma = 8.0e-11;

	m_params.t0 = 20.0;
	m_params.tj = 62.0;
	printf("Let's do AE stuff\n");
	m_params.h0 =   0.0 /100. * powf(10.0,(8.05573-1723.64/(m_params.t0+233.076))) * 133.32239;
	printf("AE1\n");
	//m_params.hj = 100.0 /100. * AE(m_params.tj);
	m_params.hj =   90.0 /100. * powf(10.0,(8.05573-1723.64/(m_params.tj+233.076))) * 133.32239;
	printf("AE2\n");

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

	printf("Try reading Z file (%d)\n",m_numVelCells.z);
	readinfile(Zfile,m_Z,m_numVelCells.z+1);
	for (int i=0;i<m_numVelCells.z;i++) {
		m_cZ[i] = (m_Z[i]+m_Z[i+1])/2.0;
	}

	//float R[gridSize.x+1];
	//float R[MAX_GRID_SIZE];
	printf("Try reading R file (%d)\n",m_numVelCells.x);
	readinfile(Rfile,m_R,m_numVelCells.x+1);
	for (int i=0;i<m_numVelCells.x;i++) {
		m_cR[i] = (m_R[i]+m_R[i+1])/2.0;
	}
    m_params.worldSize = make_float3(m_R[m_numVelCells.x]-m_R[0], 1.0f, m_Z[m_numVelCells.z]-m_Z[0]);
    m_params.cellSize = make_float3(m_params.worldSize.x / m_numCells.x, m_params.worldSize.y / m_numCells.y, m_params.worldSize.z / m_numCells.z);

	FILE *fp;
	string filename;
	string directory = "/home/rkeedy/Dropbox/CFD/TurbJetNuCoarserWide/"; // CFD directory
	//string directory = "/home/keedy/Dropbox/CFD/TurbJetNuCoarserWide/"; // CFD directory
	//string directory = "/home/user/Dropbox/CFD/TurbJetNuCoarserWide/"; // CFD directory
	string Uzfile = "UzCells.txt";
	string Urfile = "UrCells.txt";
	string nutfile = "nutCells.txt";
	string epsfile = "epsCells.txt";
	string tkefile = "kCells.txt";

	float Uz [m_numVelCells.x*m_numVelCells.z];
	float Ur [m_numVelCells.x*m_numVelCells.z];
	float tke[m_numVelCells.x*m_numVelCells.z];
	float eps[m_numVelCells.x*m_numVelCells.z];
	float nut[m_numVelCells.x*m_numVelCells.z];

	//strcpy(filename,directory);	strcat(filename,Uzfile);
	filename = directory+Uzfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in Uz\n");
	for (int i=0;i<m_numVelCells.x*m_numVelCells.z;i++) {
		fscanf(fp,"%f",&Uz[i]);
	}
	fclose(fp);

	//strcpy(filename,directory);	strcat(filename,Urfile);
	filename = directory+Urfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in Ur\n");
	for (int i=0;i<m_numVelCells.x*m_numVelCells.z;i++) {
		fscanf(fp,"%f",&Ur[i]);
	}
	fclose(fp);

	filename = directory+nutfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in nut\n");
	for (int i=0;i<m_numVelCells.x*m_numVelCells.z;i++) {
		fscanf(fp,"%f",&nut[i]);
	}
	fclose(fp);

	filename = directory+tkefile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in tke\n");
	for (int i=0;i<m_numVelCells.x*m_numVelCells.z;i++) {
		fscanf(fp,"%f",&tke[i]);
	}
	fclose(fp);

	filename = directory+epsfile;
	fp = fopen(filename.c_str(),"r");
	printf("Reading %s\n",filename.c_str());
	printf("Read in eps\n");
	for (int i=0;i<m_numVelCells.x*m_numVelCells.z;i++) {
		fscanf(fp,"%f",&eps[i]);
	}
	fclose(fp);

	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {
			m_Uz [i][k] = Uz [i*m_numVelCells.z+k];
			m_Ur [i][k] = Ur [i*m_numVelCells.z+k];
			m_eps[i][k] = eps[i*m_numVelCells.z+k];
			m_tke[i][k] = tke[i*m_numVelCells.z+k];
			m_nut[i][k] = nut[i*m_numVelCells.z+k];
			m_massdt[i][k] = 0.0;
		}
	}

	float flow_out = 0.0;
	float flow_in = 0.0;
	// RMK: do particle in/out time elapse for cells
	for (int i=0;i<m_numVelCells.x;i++) {
		for (int k=0;k<m_numVelCells.z;k++) {

			//m_volume[i][k] = (m_Z[k+1]-m_Z[k]) * m_darc/2.0 * (powf(m_R[i+1],2.0)-powf(m_R[i],2.0));
			if (i>0 && i<m_numVelCells.x-1) {
				m_gradRnut[i][k]    = (m_nut[i+1][k]-m_nut[i-1][k])/(m_R[i+1]-m_R[i-1]);
			} else {
				m_gradRnut[i][k] = 0.0;
			}
			if (k>0 && k<m_numVelCells.z-1) {
				m_gradZnut[i][k] = (m_nut[i][k+1]-m_nut[i][k-1])/(m_Z[k+1]-m_Z[k-1]);
			} else {
				m_gradZnut[i][k] = 0.0;
			}
			printf("gradnut %d %d %f %f\n",i,k,m_gradZnut[i][k],m_gradRnut[i][k]);


			if (i == m_numVelCells.x-1) {
				m_massdt[i][k] += -(m_Ur[i][k])*(m_Z[k+1]-m_Z[k])*m_params.darc*m_cR[i] * m_params.celldensity * m_params.dt;
				printf("massdt: %f %f %f\n",m_massdt[i][k],m_params.celldensity,m_params.dt);
				if (m_Ur[i][k]>0) {
					flow_out += (m_Ur[i][k])*(m_Z[k+1]-m_Z[k])*m_darc*m_R[m_numVelCells.x] * m_celldensity * m_dt;
				} else {
					flow_in  -= (m_Ur[i][k])*(m_Z[k+1]-m_Z[k])*m_darc*m_R[m_numVelCells.x] * m_celldensity * m_dt;
				}
			}
			// don't need to do i==0 because that's symmetry boundary
			if (k == 0) {
				m_massdt[i][k] += (m_Uz[i][k])*m_params.darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_params.celldensity * m_params.dt;
				if (m_Uz[i][k]>0) {
					flow_in  += (m_Uz[i][k])*m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_celldensity * m_dt;
				} else {
					flow_out -= (m_Uz[i][k])*m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_celldensity * m_dt;
				}
			} else if (k == m_numVelCells.z-1) {
				m_massdt[i][k] += -(m_Uz[i][k])*m_params.darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_params.celldensity * m_params.dt;
				if (m_Uz[i][k]>0) {
					flow_out += (m_Uz[i][k])*m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_celldensity * m_dt;
				} else {
					flow_in  -= (m_Uz[i][k])*m_darc/2.0*(powf(m_R[i+1],2)-powf(m_R[i],2)) * m_celldensity * m_dt;
				}
			}

		}
	}

    for (int i=0; i<m_numVelCells.x; i++) {
    	for (int k=0; k<m_numVelCells.z; k++) {
    	    m_params.Uz[i][k] = m_Uz[i][k];
    	    m_params.Ur[i][k] = m_Ur[i][k];
    	    m_params.nut[i][k] = m_nut[i][k];
    	    //m_params.tke[i][k] = m_tke[i][k];
    	    //m_params.eps[i][k] = m_eps[i][k];
    	    m_params.gradRnut[i][k] = m_gradRnut[i][k];
    	    m_params.gradZnut[i][k] = m_gradZnut[i][k];
    	    m_params.massdt[i][k] = m_massdt[i][k];
    	    //m_params.volume[i][k] = m_volume[i][k];
    	    m_params.massflux[i][k] = 0.0;
    	}
    }
	for (int k=0; k<=m_numVelCells.z; k++) {
	    m_params.Z[k] = m_Z[k];
	    m_params.cZ[k] = m_cZ[k];
	}
    for (int i=0; i<=m_numVelCells.x; i++) {
        m_params.R[i] = m_R[i];
        m_params.cR[i] = m_cR[i];
    }

    printf("Initializing...\n");
    //_initialize(numParticles,gridSize,deltatime);
    _initializeMax(maxNumParticles,numParticles,numVelCells,numCells,deltatime);
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
    //t = 1.0-t;
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
ParticleSystem::_initializeMax(int maxNumParticles, int numParticles, uint3 &numVelCells, uint3 &numCells, float deltaT)
{
    assert(!m_bInitialized);

    printf("In initializeMax\n");
    m_numParticles = numParticles;
    m_maxNumParticles = maxNumParticles;

    // RMK: set the size of the cylindrical domain in the y/theta direction
    m_darc = 1.0;
    m_maxParticlesPerCell = 10;
    m_minParticlesPerCell = 6;
    m_avgParticlesPerCell = (m_maxParticlesPerCell+m_minParticlesPerCell)/2;
    m_celldensity = 1.0;
    m_schmidt = 0.7;
    m_dt = deltaT;
    m_density_rho = 1.0;
    m_null_grid_value    = numCells.x*numCells.y*numCells.z+1;
    m_null_velgrid_value = numVelCells.x*numVelCells.y*numVelCells.z+1;

    m_worldOrigin = m_params.worldOrigin;
    m_numCells = numCells;
    m_numVelCells = numVelCells;

    m_minBin = 0.1;  //microns
    m_numBins = 20; //params.NUM_DIAM_BINS;
    m_binSize = 0.5; // microns
    printf("Starting a loop\n");
	for (int n=1;n<m_numBins;n++) {
		 //m_params.binDiam[n] = n*bind;
		 m_bVols[n] = 3.14159265359*powf((m_binSize*n/2.0e6),3.0) *4./3.; // # in cubic meters
	}
	printf("Ending a loop\n");
	//bindiam[0] = smallest_diam;
	m_bVols[0] = 3.14159265359*powf((m_minBin/2.0e6),3.0) *4./3.; // # in cubic meters


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
    m_hVelCellStart = new uint[m_numTotalVelCells];
    m_hCellEnd      = new uint[m_numTotalCells];
    m_hVelCellEnd   = new uint[m_numTotalVelCells];
    printf("MEMSET (cellend) %d %d\n",m_numTotalCells,m_numTotalVelCells);
    memset(m_hCellStart,    0, m_numTotalCells*sizeof(uint));
    memset(m_hVelCellStart, 0, m_numTotalVelCells*sizeof(uint));
    memset(m_hCellEnd,      0, m_numTotalCells*sizeof(uint));
    memset(m_hVelCellEnd,   0, m_numTotalVelCells*sizeof(uint));

    // RMK: these are my creations
    printf("MEMSET (particlehash)\n");
    m_hParticleHash = new uint[m_maxNumParticles];
    memset(m_hParticleHash, 0, m_maxNumParticles*sizeof(uint));
    printf("MEMSET (particleindex)\n");
    m_hParticleIndex = new uint[m_maxNumParticles];
    memset(m_hParticleIndex, 0, m_maxNumParticles*sizeof(uint));

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

    allocateArray((void **)&m_dCellStart,    m_numTotalCells*sizeof(uint));
    allocateArray((void **)&m_dVelCellStart, m_numTotalVelCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd,      m_numTotalCells*sizeof(uint));
    allocateArray((void **)&m_dVelCellEnd,   m_numTotalVelCells*sizeof(uint));
    printf("Arrays allocated\n");

    if (m_bUseOpenGL)
    {
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

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);
    //setCellParameters(&m_cellparams);

    m_bInitialized = true;

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
    return gridPosz * m_params.numVelCells.y * m_params.numVelCells.x + gridPosy * m_params.numVelCells.x + gridPosx;
}

int ParticleSystem::calcGridHashHost(int gridPosx,int gridPosy,int gridPosz)
{
	// RMK: Better hope that gridPos !> gridSize-1
    //gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    //gridPos.y = gridPos.y & (params.gridSize.y-1);
    //gridPos.z = gridPos.z & (params.gridSize.z-1);
    return gridPosz * m_params.numCells.y * m_params.numCells.x + gridPosy * m_params.numCells.x + gridPosx;
}

float ParticleSystem::volume(int j)
{
	int k = j / m_params.numCells.z;
	int i = j % m_params.numCells.x;
	return (m_Z[k+1]-m_Z[k]) * m_darc/2.0 * (powf(m_R[i+1],2.0)-powf(m_R[i],2.0));
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
	//int k = j / m_params.numCells.z;
	//int i = j % m_params.numCells.x;
    //int i = floor((posx - params.worldOrigin.x) / params.cellSize.x);
    //int j = 0;
    //int k = floor((posz - params.worldOrigin.z) / params.cellSize.z);
	//return (m_Z[k+1]-m_Z[k]) * m_darc/2.0 * (powf(m_R[i+1],2.0)-powf(m_R[i],2.0));
	// This assumes x origin is at zero (cylindrical coordinates)
	return m_params.cellSize.z * m_darc/2.0 * (powf(m_params.cellSize.x*(gridPos.x+1),2.0)-powf(m_params.cellSize.x*gridPos.x,2.0));
}

void ParticleSystem::boundaryFluxNew(//float *newVel,            // calculate and return velocities
			 float *newPos,
			 float *newScalar,
			 float *newDiamBins,
             float *sortedPos,         // accept the positions (sorted)
             float *sortedScalar,         // accept the positions (sorted)
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
	float newRad = 0.1;
	printf("np: %d\n",numParticles);

	for (int i=0;i<m_params.numVelCells.x;i++) {
		for (int k=0;k<m_params.numVelCells.z;k++) {
			gindex = calcVelGridHashHost(i,0,k);
			//printf("%d %d %d\n",i,k,gindex);
			if (i==m_params.numVelCells.x-1 || k==0 || k==m_params.numVelCells.z-1) {
				m_params.massflux[i][k] -= m_params.massdt[i][k];
				if ((m_params.massdt[i][k]/fabs(m_params.massdt[i][k]))*m_params.massflux[i][k] < 0.0) {
					//m_params.massflux[i][k] += volume(gindex)*m_params.celldensity/m_params.avgParticlesPerCell*(m_params.massdt[i][k]/fabs(m_params.massdt[i][k]));
					if (m_params.massdt[i][k]>0) {  // add a particle
						//printf("Adding part, %f\n",sortedPos[0]);
						sortedPos[numParticles*4  ] = (m_R[i+1]-m_R[i]) * frand() + m_R[i]; //p.x;
						//printf("Adding part 2\n");
						sortedPos[numParticles*4+1] = 0.0; //p.y;
						//printf("Adding part 3\n");
						sortedPos[numParticles*4+2] = (m_Z[k+1]-m_Z[k]) * frand() + m_Z[k]; //p.z;
						//printf("Adding part 4\n");
						sortedPos[numParticles*4+3] = cellVolume(calcGridPos(sortedPos[numParticles*4],0.0,sortedPos[numParticles*4+2]))*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
						m_params.massflux[i][k] += sortedPos[numParticles*4+3];
						//printf("Adding part 5\n");
						if (m_R[i+1]<m_params.jet_radius && k==0) {
							sortedScalar[numParticles] = 1.0;
						} else {
							sortedScalar[numParticles] = 0.0; //5;
						}
						sortedDiamBins[numParticles*m_params.numBins] = 1.0;
						for (int b=1; b<m_params.numBins; b++) {
							sortedDiamBins[numParticles*m_params.numBins+b] = 0.0;
						}
						numParticles += 1;
					} else { // remove a particle
						index = cellStart[gindex];
						if (index>cellEnd[gindex]) {
							printf("WE GOT A HUGE PROBLEM!............................\n");
						} else {
							m_params.massflux[i][k] -= sortedPos[index*4+3];
							sortedPos[index*4+3] = -1.0;
							lessParticles += 1;
						}
					}
				}
			}
		}
	}

	printf("Changing NP bf from %d to %d\n",numParticles,numParticles-lessParticles);

	numParticles -= lessParticles;

	uint originalIndex;
	for (int i=0;i<maxNumParticles;i++) {
	    // write new velocity back to original unsorted location
	    originalIndex = gridParticleIndex[i];
	    //newVel[originalIndex] = make_float4(vel + force, 0.0f);
	    newPos[originalIndex*4  ] = sortedPos[i*4  ];
	    newPos[originalIndex*4+1] = sortedPos[i*4+1];
	    newPos[originalIndex*4+2] = sortedPos[i*4+2];
	    newPos[originalIndex*4+3] = sortedPos[i*4+3];
	    newScalar[originalIndex]  = sortedScalar[i];
	    for (int b=0; b<m_params.numBins; b++) {
	    	newDiamBins[originalIndex*m_params.numBins+b] = sortedDiamBins[i*m_params.numBins+b];
	    }
	}

}

void ParticleSystem::balanceCells(            // calculate and return velocities
			 //float *newPos,
			 //float *newScalar,
             float *sortedPos,         // accept the positions (sorted)
			 float *sortedScalar,
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
	int partsInCell;
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
				j=0;
			} //else {
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
					//sortedScalar[cellStart[gindex]+j] = sortedScalar[numParticles];
					if (sortedPos   [numParticles         *4+3]<0) {
					printf("new cell mass: %f %d %d %d %d %f %d %d\n",sortedPos[numParticles*4+3],j, gindex, cellStart[gindex], cellEnd[gindex],sortedPos[(cellStart[gindex]+j)*4+3],(cellStart[gindex]+j),m_hCellStart[gindex]);
					}
				} else {
					sortedPos   [numParticles*4+3] = 0.0;
					sortedScalar[numParticles    ] = 0.0;
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
				if ((sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3])>0.0) {
				sortedScalar[cellStart[gindex]+j] = (sortedScalar[cellStart[gindex]+j]*sortedPos[(cellStart[gindex]+j)*4+3] + sortedScalar[cellEnd[gindex]-1]*sortedPos[(cellEnd[gindex]-1)*4+3]) /
													(sortedPos[(cellStart[gindex]+j)*4+3]+sortedPos[(cellEnd[gindex]-1)*4+3]);
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

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL)
    {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    }
    else
    {
        dPos = (float *) m_cudaPosVBO;
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
    printf("Integrating system\n");
    integrateSystem(
        dPos,
        m_dVel,
        //m_dScalar
        deltaTime,
        m_maxNumParticles);
    printf("System integrated\n");

    //copyArrayToHost(m_hPos,    dPos,      0, sizeof(float)*m_maxNumParticles*4);
    // NEW COMMENT:
    //copyArrayToHost(m_hScalar, m_dScalar, 0, sizeof(float)*m_maxNumParticles);
    //threadSync;

/*    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    printf("Check 1\n");
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    printf("Check 2\n");
    float *ptr = data;
    printf("Check 3 %f\n",m_hScalar[0]);

    for (uint i=0; i<m_maxNumParticles; i++)
    {
    	//printf("Scroll %d\n",i);
    	//printf(" is %f\n",m_hScalar[i]);
        //float t = i / (float) m_maxNumParticles;
        //colorRamp(t, ptr);

    	//colorRamp(m_hPos[i*4+3], ptr);
        colorRamp(m_hScalar[i], ptr);
        //printf("eh1 %d\n",m_numParticles);
        ptr+=3;
        //printf("eh2 %d\n",m_maxNumParticles);


        *ptr++ = 1.0f;
    }

    printf("Check 4\n");
    glUnmapBufferARB(GL_ARRAY_BUFFER);
*/

    // -----------------------
    //   PREPPING FOR FLUXES
    // -----------------------

	calcNewVelHashMax(
        m_dGridParticleHash,   // determine this based on the cell the particle is in
        m_dGridParticleIndex,  // determine index based on the thread location
        dPos,                  // use position to calc the hash
        m_maxNumParticles);
    printf("Hash calculated\n");
    // sort particles based on hash

    sortParticles(
    	m_dGridParticleHash,    // array by which to sort
    	m_dGridParticleIndex,   // array gets rearraged based on hash sorting
    	m_maxNumParticles);
    printf("Particles sorted\n");

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    // first declared in particleSystem.cuh
    printf("Reorder data\n");
    reorderDataAndFindCellStart(
        m_dVelCellStart,
        m_dVelCellEnd,
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
        m_numTotalVelCells);
    printf("Data reordered\n");


    copyArrayToHost(m_hSortedPos,     m_dSortedPos,         0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToHost(m_hSortedScalar,  m_dSortedScalar,      0, sizeof(float)*m_maxNumParticles);
    copyArrayToHost(m_hSortedDiamBins,m_dSortedDiamBins,    0, sizeof(float)*m_maxNumParticles*m_params.numBins);
    copyArrayToHost(m_hParticleIndex, m_dGridParticleIndex, 0, sizeof(uint) *m_maxNumParticles);
    copyArrayToHost(m_hParticleHash,  m_dGridParticleHash,  0, sizeof(uint) *m_maxNumParticles);
    copyArrayToHost(m_hVelCellStart,  m_dVelCellStart,      0, sizeof(uint) *m_numTotalVelCells);
    copyArrayToHost(m_hVelCellEnd,    m_dVelCellEnd,        0, sizeof(uint) *m_numTotalVelCells);

    printf("Process boundaries\n");
    //collide(
    counter = m_numParticles;
    boundaryFluxNew(
        m_hPos,
        m_hScalar,
        m_hDiamBins,
        m_hSortedPos,         // send off the sorted positions
        m_hSortedScalar,         // send off the sorted velocities
        m_hSortedDiamBins,
    	m_hParticleHash,
        m_hParticleIndex,
        m_hVelCellStart,
        m_hVelCellEnd,
        m_maxNumParticles,
        counter);
        //m_numParticles,
        //m_numTotalVelCells);
    printf("Boundaries processed %d %d\n",m_numTotalVelCells,m_numTotalCells);
    m_numParticles = counter;

    copyArrayToDevice(dPos,        m_hPos,      0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToDevice(m_dScalar,   m_hScalar,   0, sizeof(float)*m_maxNumParticles);
    copyArrayToDevice(m_dDiamBins, m_hDiamBins, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);


    // ------------------------
    //   PREPPING FOR BALANCE
    // ------------------------

    calcNewHashMax(
        m_dGridParticleHash,   // determine this based on the cell the particle is in
        m_dGridParticleIndex,  // determine index based on the thread location
        dPos,                  // use position to calc the hash
        m_maxNumParticles);
    printf("Hash calculated\n");
    // sort particles based on hash

    sortParticles(
    	m_dGridParticleHash,    // array by which to sort
    	m_dGridParticleIndex,   // array gets rearraged based on hash sorting
    	m_maxNumParticles);
    printf("Particles sorted\n");

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    // first declared in particleSystem.cuh
    printf("Reorder data\n");
    reorderDataAndFindCellStart(
            m_dCellStart,
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
            m_numTotalCells);
    printf("Data reordered\n");

    printf("Balance cells\n");
    counter = m_numParticles;

/*
    copyArrayToHost(m_hSortedPos,     m_dSortedPos,         0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToHost(m_hSortedScalar,  m_dSortedScalar,      0, sizeof(float)*m_maxNumParticles);
    copyArrayToHost(m_hParticleIndex, m_dGridParticleIndex, 0, sizeof(uint)*m_maxNumParticles);
    copyArrayToHost(m_hParticleHash,  m_dGridParticleHash,  0, sizeof(uint)*m_maxNumParticles);
    copyArrayToHost(m_hCellStart,     m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayToHost(m_hCellEnd,       m_dCellEnd, 0, sizeof(uint)*m_numGridCells);

    balanceCells(
       // m_hPos,
       // m_hScalar,
        m_hSortedPos,         // send off the sorted positions
        m_hSortedScalar,         // send off the sorted scalars
    	m_hParticleHash,
        m_hParticleIndex,
        m_hCellStart,
        m_hCellEnd,
        m_maxNumParticles,
        counter,
        //m_numParticles,
        m_numGridCells);

    //copyArrayToDevice(dPos,      m_hPos,    0, sizeof(float)*m_maxNumParticles*4);
    //copyArrayToDevice(m_dScalar, m_hScalar, 0, sizeof(float)*m_maxNumParticles);
    copyArrayToDevice(dPos,      m_hSortedPos,    0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToDevice(m_dScalar, m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles);

*/

    balanceCellsCuda(
       // m_dPos,
       // m_dScalar,
        m_dSortedPos,         // send off the sorted positions
        m_dSortedScalar,         // send off the sorted scalars
        m_dSortedDiamBins,
    	m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_maxNumParticles,
        counter,
        m_numTotalCells);

    copyArrayToHost(m_hSortedPos,    m_dSortedPos,      0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToHost(m_hSortedScalar, m_dSortedScalar,   0, sizeof(float)*m_maxNumParticles);
    copyArrayToHost(m_hDiamBins,     m_dSortedDiamBins, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);
    copyArrayToDevice(dPos,        m_hSortedPos,    0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToDevice(m_dScalar,   m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles);
    copyArrayToDevice(m_dDiamBins, m_hDiamBins,     0, sizeof(float)*m_maxNumParticles*m_params.numBins);
    //copyArrayToDevice(m_dDiamBins, m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);

    m_numParticles = counter;

    printf("Cells balanced, np %d\n",m_numParticles);

    //int prevNumParticles = m_numParticles;

	calcNewVelHashMax(
        m_dGridParticleHash,   // determine this based on the cell the particle is in
        m_dGridParticleIndex,  // determine index based on the thread location
        dPos,                  // use position to calc the hash
        m_maxNumParticles);
    printf("Hash calculated\n");
    // sort particles based on hash

    sortParticles(
    	m_dGridParticleHash,    // array by which to sort
    	m_dGridParticleIndex,   // array gets rearraged based on hash sorting
    	m_maxNumParticles);
    printf("Particles sorted 2\n");

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    // first declared in particleSystem.cuh
    printf("Reorder data\n");
    reorderDataAndFindCellStart(
            m_dVelCellStart,
            m_dVelCellEnd,
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
            m_numTotalCells);
    printf("Data reordered\n");


    // process collisions
    printf("Process advection\n");
    //collide(
    for (int i=0;i<m_maxNumParticles*3;i++) {
    	m_hRandom[i] = frand();
    }
    copyArrayToDevice(m_dRandom, m_hRandom, 0, sizeof(float)*m_maxNumParticles*3);
/*
    for (int i=0;i<m_maxNumParticles;i++) {
    	m_hRandom[i] = frand();
    }
    copyArrayToDevice(m_dRandom2, m_hRandom, 0, sizeof(float)*m_maxNumParticles);
    */
    advect(
        m_dVel,               // receive new velocities here
        m_dSortedPos,         // send off the sorted positions
        //m_dSortedVel,         // send off the sorted velocities
        m_dGridParticleIndex, // contains old order of indices
        m_dVelCellStart,
        m_dVelCellEnd,
        m_maxNumParticles,
        m_numTotalCells,
        m_dRandom);
        //m_dRandom2);
    printf("Collisions processed\n");

    // mix scalar within cells
    /*printf("Mix Curl\n");
    mixCurl(
        m_dScalar,            // receive new scalars here
        m_dSortedScalar,      // send off the sorted scalars
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);
    printf("Curl mixed\n");*/

    chemistry(
    		m_dSortedScalar,
    		m_dSortedDiamBins,
    		m_dColors,
    		m_maxNumParticles,
    		m_numParticles);

	//    copyArrayFromDevice(m_hPos, m_dSortedPos, 0, sizeof(float)*m_maxNumParticles*4);

    printf("num particles: %d\n",m_numParticles);
    //cin.ignore();
    printf("reemerge\n");

    copyArrayToHost(m_hColors,m_dColors, 0, sizeof(float)*m_maxNumParticles);
    copyArrayToHost(m_hSortedPos,m_dSortedPos, 0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToHost(m_hDiamBins,m_dSortedDiamBins, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);
    threadSync;
    copyArrayToDevice(dPos,m_hSortedPos, 0, sizeof(float)*m_maxNumParticles*4);
    copyArrayToDevice(m_dDiamBins,m_hDiamBins, 0, sizeof(float)*m_maxNumParticles*m_params.numBins);
    //changed advect so that vels would come out sorted
    threadSync;

    copyArrayToHost(m_hSortedScalar, m_dSortedScalar, 0, sizeof(float)*m_maxNumParticles);
    copyArrayToDevice(m_dScalar, m_hSortedScalar, 0, sizeof(float)*m_maxNumParticles);

    // this is only for coloring puposes:
    copyArrayToHost(m_hVel,m_dVel, 0, sizeof(float)*m_maxNumParticles*4);

    threadSync;
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);

	//printf("Check 1\n");
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	//printf("Check 2\n");
	float *ptr = data;
	//printf("Check 3 %f\n",m_hScalar[0]);

	printf("color: %f\n",m_hColors[0]);
	for (uint i=0; i<m_maxNumParticles; i++)
	{
		//colorRamp(m_hColors[i], ptr);
		//colorRamp(m_hSortedPos[i*4+3]*2000000, ptr);
		colorRamp(m_hVel[i*4]*200000, ptr);
		//colorRamp(m_hSortedScalar[i], ptr);
		ptr+=3;

		*ptr++ = 1.0f;
	}

	printf("Check 4\n");
	glUnmapBufferARB(GL_ARRAY_BUFFER);

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
    printf("reemerge2\n");

    // Try lines in here
    glBegin(GL_LINES);
    //glBegin(GL_POLYGON);
    // cube
    glColor3f(1.0, 1.0, 1.0);
    //glutWireCube(0.2);
    glVertex3f(0.0, 0.0, 0);
    glVertex3f(0.0, 0.0, 0.46);

    glVertex3f(0.0, 0.0, 0.46);
    glVertex3f(0.102, 0.0, 0.46);

    glVertex3f(0.102, 0.0, 0.46);
    glVertex3f(0.102, 0.0, 0.0);

    glVertex3f(0.102, 0.0, 0);
    glVertex3f(0.0, 0.0, 0);
/*
    glColor3f(0.5, 0.5, 0.5);
    for (int i=1;i<m_params.numVelCells.z;i++) {
    	glVertex3f(m_R[0],0.0,m_Z[i]);
    	glVertex3f(m_R[m_params.numVelCells.x],0.0,m_Z[i]);
    }
*/
    glColor3f(0.0, 0.6, 0.0);
    for (int i=1;i<m_params.numCells.z;i++) {
    	glVertex3f(m_R[0],0.0,m_params.cellSize.z*i);
    	glVertex3f(m_R[m_params.numVelCells.x],0.0,m_params.cellSize.z*i);
    }

    glEnd();


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
	int local_num_bins = 20;
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

    int local_num_bins = 20;

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
                    	//m_hPos[i*4+3] = 1.0f;  // radius
                    	m_hPos[i*4+3] = volume(calcGridHashHost(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]))*m_params.celldensity/m_params.avgParticlesPerCell;
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
	int local_num_bins = 20;

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
                    	m_hPos[p++] = cellVolume(calcGridPos(m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2]))*m_params.celldensity/m_params.avgParticlesPerCell;
                    } else {
                    	m_hPos[p++] = -1.0f;
                    }
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hScalar[s++] = 0.0f;


                    for (int b=1;b<m_params.numBins;b++) {
                    	m_hDiamBins[i*m_params.numBins+b] = 0.0;
                    }
                    m_hDiamBins[i*m_params.numBins] = 1.0;
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