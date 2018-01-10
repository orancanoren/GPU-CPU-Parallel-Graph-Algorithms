#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std;

#ifdef __cplusplus
extern"C" {
  #endif
  #include "graphio.h"
  #include "graph.h"
  #ifdef __cplusplus
}
#endif

int THR_COUNT = 32;
const int MAX_COLOR = 3584;

char gfile[2048];

typedef unsigned int uint;

void usage() {
  printf("./coloring <filename> -t=THREADS\n");
  exit(0);
}

bool conflictDetected;

void detectConflicts(const etype * row_ptr, const vtype * col_ind, const int nov, uint * colors) {
  uint conflictCounter = 0;
  static bool forbiddenArray[MAX_COLOR];
  #pragma omp threadprivate(forbiddenArray)
  #pragma omp parallel for shared(colors) num_threads(THR_COUNT) reduction(+:conflictCounter)
  for (uint vertex = 0; vertex < nov; vertex++) {
    //memset(forbiddenArray, 0, count);
    for (size_t i = 0; i < MAX_COLOR; i++) {
      forbiddenArray[i] = false;
    }
    for (uint neighborIndex = row_ptr[vertex]; neighborIndex < row_ptr[vertex + 1]; neighborIndex++) {
      const uint d1neighbor = col_ind[neighborIndex];
      forbiddenArray[colors[d1neighbor]] = true;
      for (uint d2neighborIndex = row_ptr[d1neighbor]; d2neighborIndex < row_ptr[d1neighbor + 1]; d2neighborIndex++) {
        const uint d2neighbor = col_ind[d2neighborIndex];
        if (vertex != d2neighbor) {
          forbiddenArray[colors[d2neighbor]] = true;
        }
      }
    }

    if (forbiddenArray[colors[vertex]] == true) {
      conflictDetected = true;
      conflictCounter++;
      for (size_t i = 1; i < MAX_COLOR; i++) {
        if (forbiddenArray[i] == 0) {
          colors[vertex] = i;
          break;
        }
      }
    }
  }
  cout << conflictCounter << " conflicts detected\n";
}

bool verification(const etype * row_ptr, const vtype * col_ind, const int nov, const uint * colors) {
  for (uint vertex = 0; vertex < nov; vertex++) {
    for (uint neighborInd = row_ptr[vertex]; neighborInd < row_ptr[vertex + 1]; neighborInd++) {
      const uint neighbor = col_ind[neighborInd];
      if (colors[neighbor] == colors[vertex]) {
        cout << "distance 1 conflict between " << vertex << " and " << neighbor << endl;
        return false;
      }
      for (uint d2neighborInd = row_ptr[neighbor]; d2neighborInd < row_ptr[neighbor + 1]; d2neighborInd++) {
        const uint d2neighbor = col_ind[neighborInd];
        if (colors[d2neighbor] == colors[vertex] && vertex != d2neighbor) {
          cout << "distance 2 conflict between " << vertex << " and " << d2neighbor << endl;
          return false;
        }
      }
    }
  }
  return true;
}

int colorCount(const uint * colors, int nov){
  uint max_color = 0;
  for(uint vertex = 0; vertex < nov; vertex++)
  {
    if(colors[vertex] > max_color)
    {
      max_color = colors[vertex];
    }
  }
  return max_color;
}

int main(int argc, char *argv[]) {
  omp_set_num_threads(THR_COUNT);
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
  uint * colors = new uint[nov];
  // 1 - initialize colors to 0 [0 = uncolored]
  for (size_t i = 0; i < nov; i++) {
    colors[i] = 0;
  }
  for (uint vertex = 0; vertex < nov; vertex++) {
    uint num = 1;
    for (uint neighborInd = row_ptr[vertex]; neighborInd < row_ptr[vertex + 1]; neighborInd++) {
      const uint neighbor = col_ind[neighborInd];
      if (colors[neighbor] == 0) {
        colors[neighbor] = row_ptr[vertex + 1] - neighborInd;
      }
    }
  }

  begin = std::chrono::high_resolution_clock::now();

  // distance 2 coloring begins

  // 2 - assign colors
  conflictDetected = true;
  int iterationCounter = 0;
  while (conflictDetected) {
    conflictDetected = false;
    detectConflicts(row_ptr, col_ind, nov, colors);
    ++iterationCounter;
  }
  cout << "main loop temrinated in " << iterationCounter << " loops" << endl;


  // distance 2 coloring ends

  end = std::chrono::high_resolution_clock::now();

  std::cout << "Coloring finished [" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms]" << std::endl
  << "Starting verification procedure" << std::endl;

  if (!verification(row_ptr, col_ind, nov, colors)) {
    cerr << "code doesnt work\n";
  }
  else {
    cout << "verification succeeded" << endl;
  }

  std::cout << "Coloring accuracy: " << std::endl << "Total time: "
  << std::chrono::duration_cast<std::chrono::milliseconds>(end - initial).count() << " ms" << std::endl;
  int max_color = colorCount(colors, nov);
  cout << "Color count: " << max_color << endl;
  std::cout << std::endl;

  return 0;
}
