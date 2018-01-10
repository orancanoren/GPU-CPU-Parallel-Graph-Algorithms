#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <cuda.h>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#include "graphio.h"
#include "graph.h"

#ifdef __cplusplus
}
#endif

char gfile[2048];

void usage()
{
	printf("./coloring <filename>\n");
	exit(0);
}

	/*
You can ignore the ewgths and vwghts. They are there as the read function expects those values
row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/

	/*
	-> Traditional Top-down BFS is used
	-> Perform BFS for each vertex
	-> Naive APSP (i.e. perform SSSP for all nodes)
*/

#define THREAD_PER_BLOCK 512
/*
__global__ void resetBFS(bool * d_frontier, int * d_distances, int source) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	d_frontier[tid] = false;
	d_distances[tid] = -1;
	}*/

struct BFSdata
{
	int *distances;
	bool *frontier;
};

__global__ void BFS(etype const *d_row_ptr, vtype const *d_col_ind, vtype nov, BFSdata *bfsData,
										double *d_average_distances)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x; // each thread is responsible for performing BFS to a single vertex
	if (tid < 512)
	{ // tid is a valid vertex
		double current_average_distance = 0.0;
		bool *&currentVertexFrontier = bfsData[tid].frontier;
		int *&currentDistanceArray = bfsData[tid].distances;
		bool visited_somewhere = true;

		for (int i = 0; i < nov; i++)
		{
			//printf("%d\n", currentDistanceArray[i]);
		}

		__syncthreads();

		while (visited_somewhere)
		{
			visited_somewhere = false;
			for (vtype frontierVertex = 0; frontierVertex < nov; frontierVertex++)
			{
				//printf("%d\n", frontierVertex);
				if (currentVertexFrontier[frontierVertex])
				{
					currentVertexFrontier[frontierVertex] = false;
					// frontierVertex is in frontier!
					for (int neighbor_ind = d_row_ptr[frontierVertex]; neighbor_ind < d_row_ptr[frontierVertex + 1]; frontierVertex++)
					{
						const vtype neighbor = d_col_ind[neighbor_ind];
						if (currentDistanceArray[neighbor] == -1)
						{
							currentDistanceArray[neighbor] = currentDistanceArray[frontierVertex] + 1;
							current_average_distance += static_cast<double>(currentDistanceArray[neighbor]);
							currentVertexFrontier[neighbor] = true;
							visited_somewhere = true;
						}
					}
				}
			}
		}

		d_average_distances[tid] = current_average_distance;
	}
}

void printDistances(double const *distances, unsigned int nov)
{
	for (int i = 0; i < nov; i++)
	{
		cout << "distance[" << i << "]: " << distances[i] << endl;
	}
}

void controlError(const cudaError_t cudaError, const string message = "")
{
	if (cudaError != cudaSuccess)
	{
		cout << "CUDA ERROR! " << message << endl;
		exit(1);
	}
}

int main(int argc, char *argv[])
{
	// host memory
	etype *row_ptr;
	vtype *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	vtype nov;

	chrono::high_resolution_clock::time_point begin, end, initial = std::chrono::high_resolution_clock::now();

	if (argc != 2)
		usage();

	const char *fname = argv[1];
	strcpy(gfile, fname);

	begin = std::chrono::high_resolution_clock::now();

	if (read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1)
	{
		printf("error in graph read\n");
		exit(1);
	}

	end = std::chrono::high_resolution_clock::now();

	cout << "Graph file read [" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms]" << std::endl
			 << "Starting graph coloring procedure" << std::endl;

	/****** YOUR CODE GOES HERE *******/
	cout << "Number of vertices: " << nov << endl;
	begin = std::chrono::high_resolution_clock::now();
	// set execution parameters - consider the case: nov < THREAD_PER_BLOCK
	int blockCount = 1;
	int threadCountPerBlock = nov;
	if (nov > THREAD_PER_BLOCK)
	{
		blockCount = std::ceil(static_cast<double>(nov) / THREAD_PER_BLOCK);
		threadCountPerBlock = THREAD_PER_BLOCK;
	}

	cout << "Executing the kernel with " << blockCount << " blocks with "
			 << threadCountPerBlock << " threads each" << endl;

	// host memory
	double *average_distances = new double[nov];
	for (int i = 0; i < nov; i++)
	{
		average_distances[i] = -1.0;
	}

	// device memory
	// 1 - copy row_ptr, col_ind and an array to store the results in
	cudaError_t cudaError;
	etype *d_row_ptr;
	vtype *d_col_ind;
	double *d_average_distances;
	cudaMalloc((void **)&d_row_ptr, (nov + 1) * sizeof(etype));
	cudaMalloc((void **)&d_col_ind, row_ptr[nov] * sizeof(vtype));
	cudaMalloc((void **)&d_average_distances, nov * sizeof(double));
	cudaError = cudaMemcpy(d_row_ptr, row_ptr, (nov + 1) * sizeof(etype), cudaMemcpyHostToDevice);
	controlError(cudaError, "d_row_ptr");
	cudaError = cudaMemcpy(d_col_ind, col_ind, row_ptr[nov] * sizeof(vtype), cudaMemcpyHostToDevice);
	controlError(cudaError, "d_col_ind");
	cudaError = cudaMemcpy(d_average_distances, average_distances, nov * sizeof(double), cudaMemcpyHostToDevice);
	controlError(cudaError, "d_average_distances");

	// 2 - allocate arrays that will keep frontiers and distances for each BFS
	BFSdata *d_bfsData; // device struct
	BFSdata *h_bfsData = new BFSdata[nov];
	cudaError = cudaMalloc((void **)&d_bfsData, nov * sizeof(BFSdata));
	controlError(cudaError, "d_bfsData alloc");

	cout << "Allocating and initalizing BFS data and its fields on host" << endl;
	bool **device_frontier_pointers = new bool *[nov]; // device pointers stored in array in host
	int **device_distance_pointers = new int *[nov];

	for (int i = 0; i < nov; i++)
	{

		h_bfsData[i].frontier = new bool[nov];
		h_bfsData[i].distances = new int[nov];
		for (int j = 0; j < nov; j++)
		{
			h_bfsData[i].frontier[j] = false;
			h_bfsData[i].distances[j] = -1;
		}
		h_bfsData[i].distances[i] = 0;
		h_bfsData[i].frontier[i] = true; // source vertex is in frontier initially

		// allocate device data and store a pointer to the device pointer in host
		bool *device_frontier_array_i;
		int *device_distance_array_i;
		cudaMalloc((void **)&device_frontier_array_i, nov * sizeof(bool));
		cudaMalloc((void **)&device_distance_array_i, nov * sizeof(int));
		device_frontier_pointers[i] = device_frontier_array_i;
		device_distance_pointers[i] = device_distance_array_i;
	}

	cout << "Copying host BFS data to device" << endl;
	for (int i = 0; i < nov; i++)
	{
		cudaError = cudaMemcpy(device_frontier_pointers[i], h_bfsData[i].frontier, nov * sizeof(bool), cudaMemcpyHostToDevice);
		controlError(cudaError, "frontier[" + to_string(i) + "] copy");

		cudaError = cudaMemcpy(device_distance_pointers[i], h_bfsData[i].distances, nov * sizeof(int), cudaMemcpyHostToDevice);
		controlError(cudaError, "distance[" + to_string(i) + "] copy");

		cudaError = cudaMemcpy(&(d_bfsData[i].frontier), &device_frontier_pointers[i], sizeof(d_bfsData->frontier), cudaMemcpyHostToDevice);
		controlError(cudaError, "copying frontier array " + to_string(i) + " to device struct field");

		cudaError = cudaMemcpy(&(d_bfsData[i].distances), &device_distance_pointers[i], sizeof(d_bfsData->distances), cudaMemcpyHostToDevice);
		controlError(cudaError, "copying distance array " + to_string(i) + " to device struct field");
	}
	// HtoD copy members of the struct

	cout << "Executing kernel!" << endl;
	BFS<<<blockCount, threadCountPerBlock>>>(d_row_ptr, d_col_ind, nov, d_bfsData, d_average_distances);

	cudaError = cudaMemcpy(average_distances, d_average_distances, nov * sizeof(double), cudaMemcpyDeviceToHost);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaError));

	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();

	cout << "All pairs BFS computation has been completed ["
			 << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms]" << endl;

	printDistances(average_distances, nov);
	return 0;
}
