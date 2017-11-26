#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "graphio.h"
#include "graph.h"
#include <time.h>
char gfile[2048];

void usage(){
  printf("./bfs <filename> <sourceIndex>\n");
  exit(0);
}

/*
  You can ignore the ewgths and vwghts. They are there as the read function expects those values
  row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/
/* ===============================================
  CS 406 - Parallel Computation 
  ********HOMEWORK 1***********

  The implementation uses the ideas of Top-Down and Bottom-Up BFS approach introduced 
  by Scott Beamer et. al.   

  The algorithm is essentially is as follows:
  1 - Keep an additional array of vertices for the current frontier, initialize it with source vertex
  2 - Keep the number of edges to check from frontier, the number of vertices
  in the frontier and the number of edges to check from the unvisited vertices
  3 - By computing a threshold with the variables initialized in step 2, decide
  if it is better to perform bottom up BFS or top down BFS
  4 - Before performing the preferred strategy for BFS, reset the values for nf and mf to 0
  because they will be updated as new items will be selected for the next frontier
  5 - If bottom up BFS is preferred, linear search for a vertex that is NOT visited,
  try visiting each edge of this vertex [do this in parallel] to see if it has been visited earlier, 
  if so  we have found a parent for the vertex. If there is no such vertex proceed with the next
  unvisited vertex. Update values of variables initialized in step 2
  6 - If top down BFS is preferred, create tasks to visit neighbors of each vertex in the
  frontier. Add them to the frontier after all such tasks have been finished and remove
  previous frontier vertices(not earlier!). Update values of variables initialized in step 2
  7 - If there are 0 edges to check from unvisited vertices, terminate! Otherwise go to step 3
  ================================================*/

typedef enum { false, true } bool;

// Step 1
vtype * frontier;
int frontier_tail;
int frontier_head;
int new_frontier_tail;
int mu;
int mf;
int nf;
bool visited_somewhere;
int * distances;

#define UNKNOWN -1

bool isInFrontier(const vtype v) {
  for (int i = frontier_head; i < frontier_tail; i++) {
    if (frontier[i] == v) {
      return true;
    }
  }
  return false;
}

void BottomUp(etype * row_ptr, vtype * col_ind, vtype nov) {
  // 1 - Search for an unvisited vertex in parallel
  // 2 - For each unvisited vertex, create tasks to reach neighbors and mark them
  // #pragma omp parallel for
  for (etype vertex = 0; vertex < nov; vertex++) {
    if (distances[vertex] == UNKNOWN) {
      // Unvisited vertex found
#pragma omp task
      {
	for (vtype neighbor_ind = row_ptr[vertex]; neighbor_ind < row_ptr[vertex + 1]; neighbor_ind++) {
	  const vtype neighbor = col_ind[neighbor_ind];
	  if (isInFrontier(neighbor)) {
	    // A parent for vertex i is found
	    distances[vertex] = distances[neighbor] + 1;
	    // 1 - visited vertex is added to the list of items that will be
	    // the frontiers of the next iteration
	    // 2 - Update values of mf, nf and mu
	    #pragma omp critical
	    {
	      frontier[new_frontier_tail++] = vertex;
	      mf += row_ptr[vertex + 1] - row_ptr[vertex];
	      nf += 1;
	      mu -= row_ptr[vertex + 1] - row_ptr[vertex];
	    }
	    visited_somewhere = true;
	  }
	}
      } 
    }
  }
}

bool useBottomUp() {
  double tuning_param = 14.0;
  if (mf > mu/tuning_param) {
    return true;
  }
  return false;
}

bool useTopDown(int nov) {
  double tuning_param = 24.0;
  if (nf < nov/tuning_param) {
    return true;
  }
  return false;
}

void TopDown(etype * row_ptr, vtype * col_ind, vtype nov) {
  // For each vertex in the frontier, create a task that does the following:
  // Visit each edge, mark any unvisited neighbors
  //#pragma omp parallel for
  for (int frontier_ind = frontier_head; frontier_ind < frontier_tail; frontier_ind++) {
    //#pragma omp task
    {
      const vtype frontier_vertex = frontier[frontier_ind];
#pragma omp task
      {
	for (int neighbor_ind = row_ptr[frontier_vertex]; neighbor_ind < row_ptr[frontier_vertex + 1]; neighbor_ind++) {
	  const vtype neighbor = col_ind[neighbor_ind];
	  if (distances[neighbor] == UNKNOWN) {
	    distances[neighbor] = distances[frontier_vertex] + 1;
	    // A vertex has been visited, now the following will happen:
	    // 1 - visited vertex is added to the list of items that will be
	    // the frontiers of the next iteration
	    // 2 - Update the values of mf, nf and mu
#pragma omp critical
	    {
	      frontier[new_frontier_tail++] = neighbor;
	      mf += row_ptr[neighbor + 1] - row_ptr[neighbor];
	      nf += 1;
	      mu -= row_ptr[neighbor + 1] - row_ptr[neighbor];
	    }
	    visited_somewhere = true;
	  }      
	}
      }
    }
  }
}

void BFS(etype * row_ptr, vtype * col_ind, vtype nov, const vtype source) {
  printf("BFS() invoked\n");
  etype row_1 = row_ptr[1];

  // Step 2
  mf = row_ptr[1] - row_ptr[0]; // number of edges to check from frontier
  nf = 1; // number of vertices in the frontier
  mu = row_ptr[nov] - mf - 1; // number of edges to check from unexplored vertices
  // distances are initialized to 0, this is the true value for the source vertex,
  // but for convenience let us mark it visited by setting its distance to 1 so that
  // we can use the value 0 to determine unvisited vertices

  distances[source] = 0;
  frontier[0] = source;
  frontier_head = 0;
  frontier_tail = 1;
  new_frontier_tail = 1;
  visited_somewhere = true;
  frontier[frontier_tail++] = source;

  bool topDownActive = true;
  while (visited_somewhere) { // while there exists vertices to visit
    visited_somewhere = false;
    if (topDownActive) {
      TopDown(row_ptr, col_ind, nov);
    }
    else {
      BottomUp(row_ptr, col_ind, nov);
    }
#pragma omp taskwait
    if (topDownActive) {
      topDownActive = !useBottomUp();
    }
    else {
      topDownActive = useTopDown(nov);
    }
    frontier_head = frontier_tail;
    frontier_tail = new_frontier_tail;
    mf = 0;
    nf = 0;
    }
}

int main(int argc, char *argv[]) {
  omp_set_num_threads(16);
  etype *row_ptr;
  vtype *col_ind;
  ewtype *ewghts;
  vwtype *vwghts;
  vtype nov, source;  
  
  if(argc != 3)
    usage();

  const char* fname = argv[1];
  strcpy(gfile, fname);
  source = atoi(argv[2]);
  
  if(read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1) {
    printf("error in graph read\n");
    exit(1);
  }
  
  /****** YOUR CODE GOES HERE *******/
  distances = (int*)malloc(nov*sizeof(int));
  frontier = (vtype *)malloc(nov*sizeof(vtype));
  for (int i = 0; i < nov; i++) {
    distances[i] = -1;
  }
  
  if (distances == NULL || frontier == NULL) {
    printf("Cannot allocate memory");
    exit(1);
  }

  printf("Initiating Breadth First Search\n");
  double start_time = omp_get_wtime();
  BFS(row_ptr, col_ind, nov, source);
  double time = omp_get_wtime() - start_time;
  printf("Breadth First Search completed in [%g s]\nWriting results\n", time);

  FILE *fp;
  fp = fopen("./results.txt", "w+");
  for (int i = 0; i < nov; i++) {
    fprintf(fp, "%i ", distances[i]);
  }
  fclose(fp);
  printf("results written to results.txt\n");
  
  free(row_ptr);
  free(col_ind);
    
  return 1;
}
