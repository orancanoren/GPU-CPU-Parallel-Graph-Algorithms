#include <cuda.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <climits>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

#define MAX_THREAD_PER_BLOCK 512
#define DEBUG_PRINT

/*
  PARALLEL NN - VERSION 2
*/

typedef unsigned short int usint;

const usint num_dimensions = 16;
const usint numPointsTest = 1000;
const usint numPointsTrain = 19000;
const usint streamCount = 4;

struct Coordinates {
	usint points[num_dimensions];
};

__device__ float getDistance(const Coordinates & coord1, const Coordinates & coord2) {
	float square_sum = 0;
	for (int i = 0; i < num_dimensions; i++) {
		const int c1 = coord1.points[i];
		const int c2 = coord2.points[i];
		square_sum += (c1 - c2)*(c1 - c2);
	}
	return sqrt(square_sum);
}

__global__ void nearestNeighbor(Coordinates * trainCoords, Coordinates * testCoords, const usint sizeTest, const usint sizeTrain, usint * nearestNeighbors) {
	const usint threadId = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadId < sizeTest) { // DEBUG
		usint nearestNeighbor = 0;
		usint nearestDistance = USHRT_MAX;
		for (int trainCoordInd = 0; trainCoordInd < sizeTrain; trainCoordInd++) {
			float currentDistance = getDistance(trainCoords[trainCoordInd], testCoords[threadId]);
			if (currentDistance < nearestDistance) {
				nearestNeighbor = trainCoordInd;
				nearestDistance = currentDistance;
			}
		}
		nearestNeighbors[threadId] = nearestNeighbor;
	}
}

bool checkError(const cudaError_t & error, const char * msg = "") {
	if (error != cudaSuccess) {
		printf("CUDA ERROR: %s\n", msg);
		cout << error << endl;
		exit(1);
	}
	return true;
}

int main() {
	// 1 - INITIALIZE READ STREAMS
	const char * testFile = "test.txt";
	const char * trainFile = "train.txt";
	FILE * test_is = fopen(testFile, "r"), * train_is = fopen(trainFile, "r");
	if (!test_is) {
		cerr << "Cannot open " << testFile << endl;
		exit(1);
	}
	if (!train_is) {
		cerr << "Cannot open " << trainFile << endl;
		exit(1);
	}

	cudaSetDevice(0); // initialize CUDA context
	cout << "\t--------------------\n";
	chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now(), temp, end;

	// 2 - SET EXECUTION PARAMETERS
	cudaStream_t streams[streamCount]; // create four CUDA streams
	cudaError_t cudaError;

	usint numThreadsPerBlock = numPointsTest;
	usint numBlocks = 1;
	if (numPointsTest > MAX_THREAD_PER_BLOCK) {
		numBlocks = std::ceil(static_cast<double>(numPointsTest) / MAX_THREAD_PER_BLOCK);
		numThreadsPerBlock = MAX_THREAD_PER_BLOCK;
	}
	numThreadsPerBlock /= streamCount;
	cout << "Kernels will be called with " << numBlocks << " blocks with " << numThreadsPerBlock << " threads each\n";

	// 3 - READ TRAIN COORDINATES FROM FILE STREAMS
	// device pointers
	Coordinates * d_testCoordinates[streamCount], *d_trainCoordinates;
	usint * d_nearestNeighbors[streamCount];
	// host pointers
	Coordinates * h_testCoordinates[streamCount], *h_trainCoordinates;
	usint * h_nearestNeighbors[streamCount];

	cudaError = cudaMallocHost((void**)&h_trainCoordinates, numPointsTrain * sizeof(Coordinates));
	checkError(cudaError, "cudamallochost - h_trainCoordinates");

	// read train points to host
	for (int i = 0; i < numPointsTrain; i++) {
		fscanf(train_is, "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", &h_trainCoordinates[i].points[0], &h_trainCoordinates[i].points[1], &h_trainCoordinates[i].points[2],
			&h_trainCoordinates[i].points[3], &h_trainCoordinates[i].points[4], &h_trainCoordinates[i].points[5], &h_trainCoordinates[i].points[6], &h_trainCoordinates[i].points[7],
			&h_trainCoordinates[i].points[8], &h_trainCoordinates[i].points[9], &h_trainCoordinates[i].points[10], &h_trainCoordinates[i].points[11], &h_trainCoordinates[i].points[12],
			&h_trainCoordinates[i].points[13], &h_trainCoordinates[i].points[14], &h_trainCoordinates[i].points[15]);
	}
	cout << "done reading training coordinates to host pinned memory" << endl;

	// copy train coordinates to device
	cudaError = cudaMalloc((void**)&d_trainCoordinates, numPointsTrain * sizeof(Coordinates));
	checkError(cudaError, "cudaMalloc - d_trainCoordinates");
	cudaError = cudaMemcpy(d_trainCoordinates, h_trainCoordinates, numPointsTrain * sizeof(Coordinates), cudaMemcpyHostToDevice);
	checkError(cudaError, "cudaMemcpyAsync - d_trainCoordinates");
	
	chrono::high_resolution_clock::time_point kernel_start = chrono::high_resolution_clock::now();
	for (usint stream = 0; stream < streamCount; stream++) {
		// 1 - create stream
		cudaStreamCreate(&streams[stream]);

		// 2 - Host memory - allocate memory on host for results and test coordinates
		cudaError = cudaMallocHost((void**)&h_nearestNeighbors[stream], (numPointsTest / streamCount) * sizeof(usint));
		checkError(cudaError, "cudamallochost - h_nearestneighbors");
		cudaError = cudaMallocHost((void**)&h_testCoordinates[stream], (numPointsTest / streamCount) * sizeof(Coordinates));
		checkError(cudaError, "cudamallochost - h_testCoordinates");

		// 3 - Host memory - read test points
		for (int i = 0; i < numPointsTest / streamCount; i++) {
			fscanf(test_is, "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d", &h_testCoordinates[stream][i].points[0], &h_testCoordinates[stream][i].points[1], &h_testCoordinates[stream][i].points[2],
				&h_testCoordinates[stream][i].points[3], &h_testCoordinates[stream][i].points[4], &h_testCoordinates[stream][i].points[5], &h_testCoordinates[stream][i].points[6], &h_testCoordinates[stream][i].points[7],
				&h_testCoordinates[stream][i].points[8], &h_testCoordinates[stream][i].points[9], &h_testCoordinates[stream][i].points[10], &h_testCoordinates[stream][i].points[11], &h_testCoordinates[stream][i].points[12],
				&h_testCoordinates[stream][i].points[13], &h_testCoordinates[stream][i].points[14], &h_testCoordinates[stream][i].points[15]);
		}
		
		// 4 - Device memory - allocate space for test coordiantes and result array for this stream to write its results to
		cudaError = cudaMalloc((void**)&d_testCoordinates[stream], (numPointsTest / streamCount) * sizeof(Coordinates));
		checkError(cudaError, "cudaMalloc - d_testCoordiantes");
		cudaError = cudaMalloc((void**)&d_nearestNeighbors[stream], (numPointsTest / streamCount) * sizeof(usint));
		checkError(cudaError, "cudaMalloc - d_nearestNeighbors");
	
		// 5 - copy test coordinates to device in async
		temp = chrono::high_resolution_clock::now();
		cudaError = cudaMemcpyAsync(d_testCoordinates[stream], h_testCoordinates[stream], (numPointsTest / streamCount) * sizeof(Coordinates), cudaMemcpyHostToDevice, streams[stream]);
		checkError(cudaError, "cudaMemcpy - d_testCoordinates");
		end = chrono::high_resolution_clock::now();
		cout << "data copied to device memory [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n"
			<< "executing kernel with " << numBlocks << " blocks with " << numThreadsPerBlock << " threads each" << endl;
		
		// 6 - Inovke kernel for current stream
		usint *& currentResultArray = d_nearestNeighbors[stream];
		nearestNeighbor<<< numBlocks, numThreadsPerBlock, 0, streams[stream] >>>(d_trainCoordinates, d_testCoordinates[stream], numPointsTest / streamCount, numPointsTrain, currentResultArray);
		cudaError = cudaMemcpyAsync(h_nearestNeighbors[stream], d_nearestNeighbors[stream], (numPointsTest / streamCount) * sizeof(usint), cudaMemcpyDeviceToHost, streams[stream]);
		checkError(cudaError, "cudaMemcpy - h_nearestNeighbors");
	}
	
	// Wait for GPU to terminate and fetch results
	cudaError = cudaGetLastError();
	checkError(cudaError, "before deviceSync() error!");
	cudaDeviceSynchronize();
	end = chrono::high_resolution_clock::now();
	cout << "Computation + read test data: " << chrono::duration_cast<chrono::milliseconds>(end - kernel_start).count() << " ms\n";
	cout << "\t--------------------\n";
	end = chrono::high_resolution_clock::now();
	
	ofstream os("output.txt");
	
	for (int stream = 0; stream < streamCount; stream++) {
		for (int i = 0; i < numPointsTest / streamCount; i++) {
			os << h_nearestNeighbors[stream][i] << endl;
		}
	}
	
	end = chrono::high_resolution_clock::now();
	cout << "\t--------------------\nTotal time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\nterminating\n";
	
	return 0;
}
