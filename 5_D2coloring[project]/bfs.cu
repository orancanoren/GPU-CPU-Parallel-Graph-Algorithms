#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <climits>
#include <chrono>
#include "graphio.h"
#include "graph.h"

#define THREAD_PER_BLOCK 512

char gfile[2048];

using namespace std;
typedef unsigned int uint;

void usage(){
	printf("./bfs <filename> <sourceIndex>\n");
	exit(0);
}
// CUDA STARTS

void checkCudaError(cudaError_t cudaError, const char * msg = "") {
	if (cudaError != cudaSuccess) {
		cerr << "CUDA error occured [" << msg << "]\nDescription: " << cudaGetErrorString(cudaError) << endl;
		exit(1);
	}
}


__global__
void assignColors(uint *row_ptr, int *col_ind, uint *results, int nov);

__global__
void detectConflicts(uint *row_ptr, int *col_ind, uint *results, int nov, uint *errorCode);

__global__
void printResults(const uint * results, const int nov);

/*
You can ignore the ewgths and vwghts. They are there as the read function expects those values
row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/

bool verifyResults(const uint * row_ptr, const int * col_ind, int nov, const uint * results) {
	for (int vId = 0; vId < nov; vId++) {
		for (uint d1neighborInd = row_ptr[vId]; d1neighborInd < row_ptr[vId + 1]; d1neighborInd++) {
			const uint d1neighbor = col_ind[d1neighborInd];
			if (results[vId] == results[d1neighbor]) {
				cout << "distance 1 conflict between " << vId << " and " << d1neighbor << endl;
				return false;
			}
			for (uint d2neighborInd = row_ptr[d1neighbor]; d2neighborInd < row_ptr[d1neighbor + 1]; d2neighborInd++) {
				const uint d2neighbor = col_ind[d2neighborInd];
				if (results[vId] == results[d2neighbor] && vId != d2neighbor) {
					cout << "distance 2 conflict between " << vId << " and " << d2neighbor << endl;
					return false;
				}
			}
		}
	}
	return true;
}

int main(int argc, char *argv[]) {
	cudaError_t cudaError;
	// ==== HOST MEMORY ====
	uint *row_ptr;
	int *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	int nov;
	uint * results;
	uint * errorCode = new uint;
	*errorCode = 1;

	if(argc != 2)
	usage();

	const char* fname = argv[1];
	strcpy(gfile, fname);

	if(read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1) {
		printf("error in graph read\n");
		exit(1);
	}

	results = (uint *) malloc(nov * sizeof(uint)); // will store color number for each
	for (int i = 0; i < nov; i++) {
		results[i] = 0;
	}
	// ===== DEVICE MEMORY =====
	uint *d_row_ptr;
	int *d_col_ind;
	uint *d_results;
	const size_t row_size = (nov + 1) * sizeof(uint);
	const size_t col_size = (row_ptr[nov]) * sizeof(int);
	uint * d_errorCode; // true, if conflict detected; false otherwise
	chrono::high_resolution_clock::time_point begin, temp, end;
	temp = chrono::high_resolution_clock::now();
	begin = temp;

	cudaError = cudaMalloc((void **)&d_errorCode, sizeof(uint));
	checkCudaError(cudaError, "malloc errorCode");
	cudaError = cudaMalloc((void **)&d_row_ptr, row_size);
	checkCudaError(cudaError, "malloc d_row_ptr");
	cudaError = cudaMalloc((void **)&d_col_ind, col_size);
	checkCudaError(cudaError, "malloc d_col_ind");
	cudaError = cudaMalloc((void **)&d_results, nov * sizeof(uint));
	checkCudaError(cudaError, "malloc d_results");

	cudaError = cudaMemcpy(d_results, results, nov * sizeof(uint), cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy results");
	cudaError = cudaMemcpy(d_row_ptr, row_ptr, row_size, cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy row_ptr");
	cudaError = cudaMemcpy(d_col_ind, col_ind, col_size, cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy col_ind");

	end = chrono::high_resolution_clock::now();
	cout << "HtoD copies done successfully [" << chrono::duration_cast<chrono::milliseconds>(end - temp).count() << " ms]\n";
	// ==== KERNEL LAUNCH =====
	uint numBlocks = 1;
	uint numThreadsPerBlock = nov;
	cout << "number of vertices: " << nov << endl;
	if (nov > THREAD_PER_BLOCK)
	{
		numBlocks = std::ceil(static_cast<double>(nov) / THREAD_PER_BLOCK);
		numThreadsPerBlock = THREAD_PER_BLOCK;
	}
	//const uint numBlocks = (nov + N - 1) / N;
	//const uint numThreadsPerBlock = N;

	int iterationCounter = 0; // for the following loop
	printf("running kernel with %d blocks with %d threads each\n", numBlocks, numThreadsPerBlock);
	temp = chrono::high_resolution_clock::now();
	while (*errorCode != 0) { // run kernel until no conflict occurs
		*errorCode = 0;
		cudaError = cudaMemcpy(d_errorCode, errorCode, sizeof(uint), cudaMemcpyHostToDevice);
		checkCudaError(cudaError, "memcpy errorCode");

		assignColors<<< numBlocks, numThreadsPerBlock >>>(d_row_ptr, d_col_ind, d_results, nov);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError(), "assignColors() error");

		detectConflicts<<< numBlocks, numThreadsPerBlock >>>(d_row_ptr, d_col_ind, d_results, nov, d_errorCode);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError(), "detectConflicts() error");

		cudaError = cudaMemcpy(errorCode, d_errorCode, sizeof(uint), cudaMemcpyDeviceToHost);
		checkCudaError(cudaError);

		iterationCounter++;
		//cout << "iteration " << iterationCounter << " is over\n";
		cout << iterationCounter << " num of fixes " << *errorCode << "\n";
	}
	end = chrono::high_resolution_clock::now();
	cout << iterationCounter << " iterations passed ["
	<< chrono::duration_cast<chrono::milliseconds>(end - temp).count() <<  "ms]\n";
	//printResults<<< numBlocks, numThreadsPerBlock >>>(d_results, nov);
	//cudaDeviceSynchronize();
	//checkCudaError(cudaGetLastError(), "printResults() error");
	cudaError = cudaMemcpy(results, d_results, nov*sizeof(uint), cudaMemcpyDeviceToHost);

	checkCudaError(cudaError, "DtoH memcpy results");

	// TODO use results
	if (!verifyResults(row_ptr, col_ind, nov, results)) {
		cout << "fonksiyonun icinde soylemicem" << endl;
	}
	uint max = 0;
	for (size_t i = 0; i < nov; i++) {
		if (results[i] > max) {
			max = results[i];
		}
	}
	std::cout << "max color is " << max << '\n';
	cudaFree(d_row_ptr);
	cudaFree(d_col_ind);
	cudaFree(d_results);

	free(row_ptr);
	free(col_ind);

	return 1;
}
