#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include "coloring.hpp"

#ifdef __cplusplus
extern"C" {
#endif
#include "graphio.h"
#include "graph.h"
#ifdef __cplusplus
}
#endif

char gfile[2048];

void usage() {
	printf("./coloring <filename> -t=THREADS\n");
	exit(0);
}

/*
  You can ignore the ewgths and vwghts. They are there as the read function expects those values
  row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/

/* ====== THE ALGORITHM ========
	The algorithm implemented in this project follows the methodologies proposed by Deveci et al.
					" Parallel Graph Coloring for Manycore Architectures
 */

int main(int argc, char *argv[]) {
	etype *row_ptr;
	vtype *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	vtype nov;

	std::chrono::high_resolution_clock::time_point begin, end, initial = std::chrono::high_resolution_clock::now();

	if (argc < 2)
		usage();

	const char* fname = argv[1];
	strcpy(gfile, fname);

	begin = std::chrono::high_resolution_clock::now();

	if (read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1) {
		printf("error in graph read\n");
		exit(1);
	}

	end = std::chrono::high_resolution_clock::now();

	std::cout << "Graph file read [" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms]" << std::endl
		<< "Starting graph coloring procedure" << std::endl;

	/****** YOUR CODE GOES HERE *******/
	GraphColoring coloring(row_ptr, col_ind, nov);
	
	begin = std::chrono::high_resolution_clock::now();
	coloring.perform_coloring();
	end = std::chrono::high_resolution_clock::now();

	std::cout << "Coloring finished [" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms]" << std::endl
		<< "Starting accuracy computation procedure" << std::endl;

	double accuracy = coloring.accuracy()*100;
	
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Coloring accuracy: " << accuracy << "%" << std::endl << "Total time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - initial).count() << " ms" << std::endl;

	std::cout << std::endl;

	return 0;
}
