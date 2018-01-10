#include <stdio.h>
#include "graph.h"

typedef unsigned int uint;

__global__
void assignColors(uint *row_ptr, int *col_ind, uint *results, int nov) {
	const uint vIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIndex == 0) {
		//printf("row_ptr[nov]: %d\n", row_ptr[nov]);
	}
	if (vIndex < nov && results[vIndex] == 0) {
		uint selectedColor = 1;
		// traverse d1 and d2 neighbors to check if selectedColor is available
		for (uint neighborIndex = row_ptr[vIndex]; neighborIndex < row_ptr[vIndex + 1]; neighborIndex++) { // distance-1 neighbor loop
			uint d1neighbor = col_ind[neighborIndex];
			if (selectedColor == results[d1neighbor]) {
				selectedColor++;
				neighborIndex = row_ptr[vIndex] - 1; // reset the loop
				continue;
			}
			for (uint d2neighborIndex = row_ptr[d1neighbor]; d2neighborIndex < row_ptr[d1neighbor + 1]; d2neighborIndex++) {
				uint d2neighbor = col_ind[d2neighborIndex];
				if (selectedColor == results[d2neighbor]) {
					selectedColor++; // ege'yle dun bunu unutmusuz :(
					neighborIndex = row_ptr[vIndex] - 1; // reset the loop
					break;
				}
			}
		}
		results[vIndex] = selectedColor; // ege'yle bunu da unutmusuz :((
	}
}
__global__
void printResults(const uint * results, const int nov) {
	const uint vIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIndex == 0) {
		printf("printResults() invoked\n");
		for (int i = 0; i < nov; i++) {
			printf("%d ", results[i]);
		}
		printf("\n");
	}
}

// && results[vIndex] != 0
__global__
void detectConflicts(uint *row_ptr, int *col_ind, uint *results, int nov, uint * errorCode) {
	const uint vIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIndex == 0) {
		//printf("detectConflicts() invoked\n");
	}
	if (vIndex < nov) {
		bool fixDone = false;
		bool secondDistanceConflictFound = false;
		for (uint neighborIndex = row_ptr[vIndex]; neighborIndex < row_ptr[vIndex+1]
			&& !secondDistanceConflictFound && results[vIndex] != 0; neighborIndex++) { // distance-1 neighbor loop
			const uint d1neighbor = col_ind[neighborIndex];
			if (results[vIndex] == results[d1neighbor]) {
				if (vIndex < d1neighbor) {
					//printf("%d - %d collision - resetting %d\n", vIndex, d1neighbor, vIndex);
					atomicMax(&results[vIndex], results[vIndex]+1);
				}
				else {
					//printf("%d - %d collision - resetting %d", vIndex, d1neighbor, d1neighbor);
					atomicMax(&results[d1neighbor], results[d1neighbor]+1);
				}
				fixDone = true;
				break;
			}
			for (uint d2neighborIndex = row_ptr[d1neighbor]; d2neighborIndex < row_ptr[d1neighbor + 1]
				&& !secondDistanceConflictFound; d2neighborIndex++) {
				const uint d2neighbor = col_ind[d2neighborIndex];
				if (results[vIndex] == results[d2neighbor] && vIndex != d2neighbor) {
					if (vIndex < d2neighbor) {
						//printf("%d - %d collision - resetting %d\n", vIndex, d2neighbor, vIndex);
						atomicMax(&results[vIndex], results[vIndex]+1);
					}
					else {
						//printf("%d - %d collision - resetting %d\n", vIndex, d2neighbor, d2neighbor);
						atomicMax(&results[d2neighbor], results[d2neighbor]+1);
					}
					fixDone = true;
					secondDistanceConflictFound = true;
				}
			}
		}
		if (fixDone) {
			atomicAdd(errorCode, 1);
		}
	}
}
