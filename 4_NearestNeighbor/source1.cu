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
  PARALLEL NN - VERSION 1
*/

typedef unsigned short int uint;

const uint num_dimensions = 16;

struct Coordinates {
	uint points[num_dimensions];
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

__global__ void nearestNeighbor(Coordinates * trainCoords, Coordinates * testCoords, const uint sizeTest, const uint sizeTrain, uint * nearestNeighbors) {
	const uint threadId = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadId < sizeTest) {
		uint nearestNeighbor = 0;
		uint nearestDistance = UINT_MAX;
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
		printf("CUDA ERROR\n");
		printf("%s\n", msg);
		return false;
	}
	return true;
}

void initReadStreams(ifstream & test_is, ifstream & train_is, Coordinates * h_trainData, Coordinates * h_testData) {
	// 1 - initialize the read streams
	const char * testFile = "test.txt";
	const char * trainFile = "train.txt";
	test_is.open(testFile);
	if (!test_is.is_open()) {
		cerr << "Cannot open " << testFile << endl;
		exit(1);
	}
	train_is.open(trainFile);
	if (!train_is.is_open()) {
		cerr << "Cannot open " << trainFile << endl;
		exit(1);
	}

	// 2 - read the training data
	string buffer;
	uint iter = 0;
	while (getline(train_is, buffer)) {
		int prev_index = 0;
		int comma_index = buffer.find(',');
		for (int i = 0; i < num_dimensions; i++) {
			h_trainData[iter].points[i] = atoi(buffer.substr(prev_index, comma_index).c_str());
			prev_index = comma_index + 1;
			comma_index = buffer.find(',', prev_index);
		}
		iter++;
	}
	train_is.seekg(0);

	// 3 - read the test data
	iter = 0;
	while (getline(test_is, buffer)) {
		uint prev_index = 0;
		uint comma_index = buffer.find(',');
		Coordinates current_coordinate;
		for (int i = 0; i < num_dimensions; i++) {
			current_coordinate.points[i] = atoi(buffer.substr(prev_index, comma_index).c_str());
			prev_index = comma_index + 1;
			comma_index = buffer.find(',', prev_index);
		}
		h_testData[iter++] = current_coordinate;
	}
	test_is.seekg(0);
}

int main() {
	cout << "\t--------------------\n";
	const uint numPointsTest = 1000;
	const uint numPointsTrain = 19000;
	chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now(), temp, end;
	// set execution parameters
	uint numThreadsPerBlock = numPointsTest;
	uint numBlocks = 1;
	if (numPointsTest > MAX_THREAD_PER_BLOCK) {
		numBlocks = std::ceil(static_cast<double>(numPointsTest) / MAX_THREAD_PER_BLOCK);
		numThreadsPerBlock = MAX_THREAD_PER_BLOCK;
	}

	temp = chrono::high_resolution_clock::now();
	cudaError_t cudaError;
	// Host memory - allocate pinned memory for train and test coordinates
	Coordinates * h_testCoordinates, *h_trainCoordinates;
	uint * h_nearestNeighbors;
	cudaError = cudaMallocHost((void**)&h_testCoordinates, numPointsTest * sizeof(Coordinates));
	checkError(cudaError, "cudamallochost - h_testCoordinates");
	cudaError = cudaMallocHost((void**)&h_trainCoordinates, numPointsTrain * sizeof(Coordinates));
	checkError(cudaError, "cudamallochost - h_trainCoordinates");
	cudaError = cudaMallocHost((void**)&h_nearestNeighbors, numPointsTest * sizeof(uint));
	checkError(cudaError, "cudamallochost - h_nearestneighbors");

	end = chrono::high_resolution_clock::now();
	cout << "allocated pinned memory on host [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";
	
	temp = chrono::high_resolution_clock::now();
	// fill host memory
	ifstream test_in, train_in;
	initReadStreams(test_in, train_in, h_trainCoordinates, h_testCoordinates);
	end = chrono::high_resolution_clock::now();
	cout << "host memory filled [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";

	// Device memory - allocate memory for train and test coordiantes on VRAM
	temp = chrono::high_resolution_clock::now();
	Coordinates * d_testCoordinates, *d_trainCoordinates;
	uint * d_nearestNeighbors;
	cudaError = cudaMalloc((void**)&d_testCoordinates, numPointsTest * sizeof(Coordinates));
	checkError(cudaError, "cudaMalloc - d_testCoordiantes");
	cudaError = cudaMalloc((void**)&d_trainCoordinates, numPointsTrain * sizeof(Coordinates));
	checkError(cudaError, "cudaMalloc - d_trainCoordinates");
	cudaError = cudaMalloc((void**)&d_nearestNeighbors, numPointsTest * sizeof(uint));
	checkError(cudaError, "cudaMalloc - d_nearestNeighbors");
	end = chrono::high_resolution_clock::now();

	cout << "allocated memory on device[" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";

	// Device memory - fill device memory
	temp = chrono::high_resolution_clock::now();
	cudaError = cudaMemcpy(d_testCoordinates, h_testCoordinates, numPointsTest * sizeof(Coordinates), cudaMemcpyHostToDevice);
	checkError(cudaError, "cudaMemcpy - d_testCoordinates");
	cudaError = cudaMemcpy(d_trainCoordinates, h_trainCoordinates, numPointsTrain * sizeof(Coordinates), cudaMemcpyHostToDevice);
	checkError(cudaError, "cudaMemcpy - d_trainCoordinates");
	end = chrono::high_resolution_clock::now();

	cout << "device memory filled [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n"
		<< "executing kernel with " << numBlocks << " blocks with " << numThreadsPerBlock << " threads each" << endl;
	temp = chrono::high_resolution_clock::now();
	// Execute kernel
	cout << "\t--------------------\n";
	nearestNeighbor <<<numBlocks, numThreadsPerBlock >>> (d_trainCoordinates, d_testCoordinates, numPointsTest, numPointsTrain, d_nearestNeighbors);
	// Wait for GPU to terminate and fetch results
	cudaDeviceSynchronize();
	end = chrono::high_resolution_clock::now();
	cout << "Kernel execution completed [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";
	
	temp = chrono::high_resolution_clock::now();
	cudaError = cudaMemcpy(h_nearestNeighbors, d_nearestNeighbors, numPointsTest * sizeof(uint), cudaMemcpyDeviceToHost);
	checkError(cudaError, "cudaMemcpy - h_nearestNeighbors");
	end = chrono::high_resolution_clock::now();
	cout << "result array copied from device to host successfully [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";

	ofstream os("output.txt");
	for (int i = 0; i < numPointsTest; i++) {
		os << h_nearestNeighbors[i] << endl;
	}
	end = chrono::high_resolution_clock::now();
	cout << "\t--------------------\nTotal time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\nterminating\n";
	
	return 0;
}
