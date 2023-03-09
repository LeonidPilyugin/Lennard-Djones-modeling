// File contains program, modelling real gas
// Leonid Pilyugin <l.pilyugin04@gmail.com>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FACTOR 2e7f
// block size
#define BLOCK_SIZE 256
// number of particles
// ~20e3
#define NN 32
//64
#define NNF ((float) NN)
#define N (NN * NN * NN)
// distance epsilon
#define EPS 1e-12f
// time step
// уменьшить в 10 раз
#define DT (1e-3f * SIGMA / VELOCITY)
//(5e-4f * SIZE / VELOCITY)
// output file name
#define OUTPUT "output.txt"
// number of steps
#define STEPS 7000000
#define SKIP 2000
#define K (1.380649e-23f * FACTOR * FACTOR)
// gas interaction parameter
#define EPSILON (6.03f * K)
// gas interaction parameter
#define SIGMA (2.63e-10f * FACTOR)
#define VELOCITY (1.5e2f * FACTOR)
#define MASS 6.645e-27f
#define SIZE 1.0f
#define PI 3.1415926f
#define MAX_VELOCITY (10.0f * VELOCITY)
#define BINS 1024
#define HIST_WIDTH (MAX_VELOCITY / BINS)


// host arrays for positions and velocities of all particles for 1 step
float4 *p, *v, *pHost, *vHost;
float *impuls, *impulsDevice;
// variables for energy and impuls computing
float *energy, *energyDevice;
// arrays for hist computing
unsigned int *counts, *countsDevice;

// Moves particles
// 
// Arguments:
//     float4 *newPos -- array of new positions
//     float4 *newVel -- array of new velocities
//     float4 *oldPos -- array of old positions
//     float4 *oldVel -- array of old velocities
//     float  dt      -- time interval
// 
__global__ void moveBodies(float4 *newPos, float4 *newVel, float4 *oldPos, float4 *oldVel, float dt) {
    // thread index
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    // position of current particle
    float4 pos = oldPos[index];
    // force
    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    // distance to other particle
    float3 r;
    // velocity
    float4 vel;
    float invDist, s, r2;
    unsigned long ind = 0;
    // shared buffer
    __shared__ float4 sp[BLOCK_SIZE];

    for (unsigned long i = 0; i < N / BLOCK_SIZE; i++, ind += BLOCK_SIZE) {
        // fill shared buffer
        sp[threadIdx.x] = oldPos[ind + threadIdx.x];
        __syncthreads();

        // compute forces from other particles
        for (unsigned long j = 0; j < BLOCK_SIZE; j++) {
            // compute distances
            r.x = pos.x - sp[j].x - __float2int_rn((pos.x - sp[j].x) / SIZE) * SIZE;
            r.y = pos.y - sp[j].y - __float2int_rn((pos.y - sp[j].y) / SIZE) * SIZE;
            r.z = pos.z - sp[j].z - __float2int_rn((pos.z - sp[j].z) / SIZE) * SIZE;
            // compute inverted distance
            r2 = r.x * r.x + r.y * r.y + r.z * r.z;

            if (r2 > 36.0f * SIGMA * SIGMA) {
                continue;
            }

            invDist = 1.0f / sqrtf(r2 + EPS) * r2 / (r2 + EPS);
            // compute force
            s = invDist * 24.0f * EPSILON * (2.0f * powf(SIGMA * invDist, 12.0f) - powf(SIGMA * invDist, 6.0f));
            f.x += r.x * s * invDist;
            f.y += r.y * s * invDist;
            f.z += r.z * s * invDist;

        }
        __syncthreads();
    }

    vel = oldVel[index];
    // accelerate
    vel.x += f.x * dt / MASS;
    vel.y += f.y * dt / MASS;
    vel.z += f.z * dt / MASS;
    // move
    pos.x = fmod(SIZE + pos.x + vel.x * dt, SIZE);
    pos.y = fmod(SIZE + pos.y + vel.y * dt, SIZE);
    pos.z = fmod(SIZE + pos.z + vel.z * dt, SIZE);
    // write answer
    newPos[index] = pos;
    newVel[index] = vel;
}

// 
__global__ void computeHistDevice(float4 *v, unsigned int *counts) {
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    float4 _vel = v[index];
    float vel = sqrtf(_vel.x * _vel.x + _vel.y * _vel.y + _vel.z * _vel.z);
    unsigned int ind = __float2uint_rd(vel / HIST_WIDTH);
    atomicAdd(counts + ind, 1);
}

void computeHist() {
    for (unsigned int i = 0; i < BINS; i++) {
        counts[i] = 0;
    }
    cudaMemcpy(countsDevice, counts, BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    computeHistDevice<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(v, countsDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(counts, countsDevice, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

// Computes total energy of system
// 
// Arguments:
//     float4 *p -- array of positions
//     float4 *v -- array of velocities
//     float *result -- result array
// 
__global__ void computeEnergyDevice(float4 *p, float4 *v, float *energy, float *impuls) {
    // thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // shared buffer
    __shared__ float4 sp[BLOCK_SIZE];
    // distance to other particles
    float3 r;
    // position of current particle
    float4 pos = p[index];
    // velocity of current particle
    float4 vel = v[index];
    float invDist, r2, ke;
    long ind = 0;

    for (long i = 0; i < N / BLOCK_SIZE; i++, ind += BLOCK_SIZE) {
        // fill shared buffer
        sp[threadIdx.x] = p[ind + threadIdx.x];
        __syncthreads();

        // compute energies from other particles
        for (long j = 0; j < BLOCK_SIZE; j++) {
            // compute distances
            r.x = pos.x - sp[j].x - __float2int_rn((pos.x - sp[j].x) / SIZE) * SIZE;
            r.y = pos.y - sp[j].y - __float2int_rn((pos.y - sp[j].y) / SIZE) * SIZE;
            r.z = pos.z - sp[j].z - __float2int_rn((pos.z - sp[j].z) / SIZE) * SIZE;
            // compute inverted distance
            r2 = r.x * r.x + r.y * r.y + r.z * r.z;

            invDist = 1.0f / sqrtf(r2 + EPS) * r2 / (r2 + EPS);
            // compute force
            atomicAdd(energy, 2.0f * EPSILON * (powf(SIGMA * invDist, 12.0f) - powf(SIGMA * invDist, 6.0f)));

        }
        __syncthreads();
    }

    ke = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) * MASS / 2.0f;
    // return total energy of this particle
    atomicAdd(energy, ke);
    atomicAdd(energy + 1, ke);
    atomicAdd(impuls, vel.x);
    atomicAdd(impuls + 1, vel.y);
    atomicAdd(impuls + 2, vel.z);
}

// 
void computeEnergyImpuls() {
    energy[0] = energy[1] = 0.0f;
    impuls[0] = impuls[1] = impuls[2] = 0.0f;
    // create array on device
    cudaMemcpy(energyDevice, energy, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(impulsDevice, impuls, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // compute energy
    computeEnergyDevice<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(p, v, energyDevice, impulsDevice);
    cudaDeviceSynchronize();

    // copy result
    cudaMemcpy(energy, energyDevice, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(impuls, impulsDevice, 3 * sizeof(float), cudaMemcpyDeviceToHost);
}


// Initializes a array with values from [-ratio/2; ratio/2]
// 
// Arguments:
//     float4 *a    -- array to initialize
//     float  ratio -- float coefficient
// 
void randomInit(float4 *a, float ratio) {
    float ximp = 0.0f;
    float yimp = 0.0f;
    float zimp = 0.0f;
    for (long i = 0; i < N; i++) {
        a[i].x = ratio * (random() / (float) RAND_MAX - 0.5f);
        a[i].y = ratio * (random() / (float) RAND_MAX) - 0.5f;
        a[i].z = ratio * (random() / (float) RAND_MAX - 0.5f);
        ximp += a[i].x;
        yimp += a[i].y;
        zimp += a[i].z;
    }

    ximp /= N;
    yimp /= N;
    zimp /= N;

    for (long i = 0; i < N; i++) {
        a[i].x -= ximp;
        a[i].y -= yimp;
        a[i].z -= zimp;
    }
}

void coordinateInit(float4 *a, float ratio) {
    for (long i = 0; i < N; i++) {
        a[i].x = ratio * ((i % NN) / NNF);
        a[i].y = ratio * (((i / NN) % NN) / NNF);
        a[i].z = ratio * (((i / NN / NN) % NN) / NNF);
    }
}

void saveInfo(FILE *output) {
    fprintf(output, "%d\n", STEPS);
    fprintf(output, "%d\n", SKIP);
    fprintf(output, "%d\n", N);
    fprintf(output, "%e\n", DT);
    fprintf(output, "%e\n", FACTOR);
    fprintf(output, "%e\n", EPSILON);
    fprintf(output, "%e\n", SIGMA);
    fprintf(output, "%e\n", VELOCITY);
    fprintf(output, "%e\n", MASS);
    fprintf(output, "%e\n", SIZE);
    fprintf(output, "%e\n", MAX_VELOCITY);
    fprintf(output, "%e\n", HIST_WIDTH);
    fprintf(output, "%d\n", BINS);
    fprintf(output, "\n");
    fflush(output);
}

// Saves frame to text file
// 
// Arguments:
//     FILE *output -- output file
//     
void saveFrame(FILE *output) {
    computeEnergyImpuls();
    // compute hist
    computeHist();
    // save energy
    fprintf(output, "%f %f\n%f %f %f\n", energy[0], energy[1], impuls[0], impuls[1], impuls[2]);

    for (unsigned int i = 0; i < BINS; i++) {
        fprintf(output, "%d ", counts[i]);
    }

    fprintf(output, "\n\n");

    fflush(output);
}

void savePositions(FILE *output) {
    fprintf(output, "%d\n\n", N);
    cudaMemcpy(pHost, p, N * sizeof(float4), cudaMemcpyDeviceToHost);
    for (long i = 0; i < N; i++) {
        fprintf(output, "He\t%f\t%f\t%f\n", pHost[i].x, pHost[i].y, pHost[i].z);
    }
    fflush(output);
}

int main(int argc, char *argv[]) {
    // seed random
    srandom(0);
    // open output file
    FILE *output = fopen(OUTPUT, "w");
    FILE *positions = fopen("positions.xyz", "w");
    // save configuration
    saveInfo(output);

    // allocate energy and counts arrays on host
    energy = (float*) malloc(2 * sizeof(float));
    impuls = (float*) malloc(3 * sizeof(float));
    counts = (unsigned int*) malloc(BINS * sizeof(unsigned int));
    // allocate energy and count arrays on device
    cudaMalloc(&energyDevice, 2 * sizeof(float));
    cudaMalloc((void**) &impulsDevice, 3 * sizeof(float));
    cudaMalloc(&countsDevice, BINS * sizeof(unsigned int));

    // allocate p and v arrays on host
    pHost = (float4*) malloc(N * sizeof(float4));
    vHost = (float4*) malloc(N * sizeof(float4));

    // init positions
    coordinateInit(pHost, SIZE);
    // init velocities
    randomInit(vHost, VELOCITY);

    // arrays for computing on device
    float4 *pps[2], *vps[2];
    // init arrays
    cudaMalloc(&(pps[0]), N * sizeof(float4));
    cudaMalloc(&(pps[1]), N * sizeof(float4));
    cudaMalloc(&(vps[0]), N * sizeof(float4));
    cudaMalloc(&(vps[1]), N * sizeof(float4));

    // start and stop events
    cudaEvent_t start, stop;
    // create start and stop events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // calculation time
    float calcTime = 0.0f;
    
    // start time
    cudaEventRecord(start, 0);

    // process

    // copy start positions from host to device
    cudaMemcpy(pps[0], pHost, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(vps[0], vHost, N * sizeof(float4), cudaMemcpyHostToDevice);

    // start time for output
    clock_t end, begin = clock();

    float elapsed, progress;
    cudaDeviceSynchronize();

    for (unsigned int i = 0, j = 0; i < STEPS + 1; i++, j = 1 - j) {
        // step
        moveBodies<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(pps[1-j], vps[1-j], pps[j], vps[j], DT);
        cudaDeviceSynchronize();

        // write to file
        if (STEPS / 1000 > 0 && i % (STEPS / 1000) == 0) {
            end = clock();
            elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
            progress = (i + 1.0f) / (STEPS + 1);

            printf("Elapsed time: %.2f s\tprogress: %.1f%% \tleft: %.1f s\n", elapsed, progress * 100.0f, elapsed / progress * (1 - progress));
            fflush(stdout);
        }

        if (i % SKIP == 0) {
            p = pps[1-j];
            v = vps[1-j];
            saveFrame(output);
            savePositions(positions);
        }
    }

    // stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // elapsed time
    cudaEventElapsedTime(&calcTime, start, stop);

    // print elapsed time
    printf("Elapsed time: %.2f\n", calcTime);

    // close output
    fclose(output);
    fclose(positions);

    // free device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(pps[0]);
    cudaFree(pps[1]);
    cudaFree(vps[0]);
    cudaFree(vps[1]);
    cudaFree(energyDevice);
    cudaFree(impulsDevice);
    cudaFree(countsDevice);

    // free host memory
    free(pHost);
    free(vHost);
    free(energy);
    free(impuls);
    free(counts);
}
