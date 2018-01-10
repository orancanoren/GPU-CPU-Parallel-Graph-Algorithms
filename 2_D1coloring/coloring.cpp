#include "coloring.hpp"
#include <exception>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <forward_list>

using namespace std;

typedef unsigned int uint;

#define UNKNOWN 0

// ========================================================
// CLASS GraphColoring - Public Member Function Definitions
// ========================================================

GraphColoring::GraphColoring(etype * row_ptr, vtype* col_ind, int nov) {
	num_vertices = nov;
	num_colored = 0;
	this->row_ptr = row_ptr;
	this->col_ind = col_ind;
	colors = new unsigned int[nov];
	unvisited_vertices.resize(nov);
	for (int i = 0; i < nov; i++) {
	  unvisited_vertices[i] = i;
	}
	unvisited_vertices_tail = nov;
	
	for (int i = 0; i < nov; i++) {
	  colors[i] = 0;
	}
	
	forbiddens.resize(nov);

	if (colors == nullptr) {
		throw std::bad_alloc();
	}
}

GraphColoring::~GraphColoring() {
	delete[] row_ptr;
	delete[] col_ind;
}

bool GraphColoring::allColored() const {
  bool allIsColored = true;
  uint num_colored = 0;
#pragma omp parallel for shared(allIsColored) reduction(+:num_colored)
  for (int i = 0; i < num_vertices; i++) {
    if (colors[i] == UNKNOWN) {
      allIsColored = false;
    }
    num_colored++;
  }
  return allIsColored;
}

bool GraphColoring::accuracy() const {
	int false_colored = 0;
#pragma omp parallel for reduction(+:false_colored)
	for (int v_i = 0; v_i < num_vertices; v_i++) {
		for (int neighbor_ind = row_ptr[v_i]; neighbor_ind < row_ptr[v_i + 1]; neighbor_ind++) {
		  const int neighbor = col_ind[neighbor_ind];
			if (colors[neighbor] == colors[v_i]) {
				false_colored++;
			}
		}
	}
	false_colored /= 2;
	
	int color_with_max_id = 0;
#pragma omp parallel
	{
		int t_max = -1;
#pragma omp for
		for (int i = 0; i < num_vertices; i++) {
			t_max = std::max(t_max, static_cast<int>(colors[i]));
		}
#pragma omp critical
		{
			color_with_max_id = std::max(t_max, color_with_max_id);
		}
	}
	std::cout << color_with_max_id << " color were used" << std::endl;
	cout << "neighbors having same colors: " << false_colored << std::endl;
	return ((num_vertices - false_colored) / static_cast<double>(num_vertices));
}

void GraphColoring::printColors() const {
	for (int i = 0; i < num_vertices; i++) {
		cout << "color " << i << ": " << colors[i] << endl;
	}
}

void GraphColoring::perform_coloring() {
	uint num_iterations = 0;
	bool coloringFinished = false;
	while (!coloringFinished) {
	  //step_beg = chrono::high_resolution_clock::now();
	  assignColors();
	  //step_end = chrono::high_resolution_clock::now();
	  //cout << "assigned colors in " << chrono::duration_cast<chrono::milliseconds>(step_end - step_beg).count() << " ms" << endl;
#pragma omp barrier
	  //step_beg = chrono::high_resolution_clock::now();
	  detectConflicts();
	  //step_end = chrono::high_resolution_clock::now();
	  //cout << "detected conflicts in " << chrono::duration_cast<chrono::milliseconds>(step_end - step_beg).count() << " ms" << endl;
#pragma omp barrier
	  //step_beg = chrono::high_resolution_clock::now();
	  forbidColors();
	  //step_end = chrono::high_resolution_clock::now();
	  //cout << "forbid colors in " << chrono::duration_cast<chrono::milliseconds>(step_end - step_beg).count() << " ms" << endl;
#pragma omp barrier
	  num_iterations++;
	  coloringFinished = allColored();
	  cout << "ITERATION " << num_iterations << endl;
	}
}

// ========================================================
// CLASS GraphColoring - Private Member Function Definitions
// ========================================================

void GraphColoring::assignColors() {
#pragma omp parallel for
  for (uint vertex_ind = 0; vertex_ind < unvisited_vertices_tail; vertex_ind++) {
    const uint vertex = unvisited_vertices[vertex_ind];
    colors[vertex] = getNextColor(vertex);		
  }
}

void GraphColoring::detectConflicts() {
	// Traverse edges of the graph, for edges (u, v) where C[u] = C[v] reset color of min(u, v)
#pragma omp parallel for schedule(guided)
  for (unsigned int vertex_ind = 0; vertex_ind < unvisited_vertices_tail; vertex_ind++) {
    const uint vertex = unvisited_vertices[vertex_ind];
		for (unsigned int neighbor_ind = row_ptr[vertex]; neighbor_ind < row_ptr[vertex + 1]; neighbor_ind++) {
		  const uint neighbor_label = col_ind[neighbor_ind];
		    if (colors[neighbor_label] == colors[vertex]) {
		      const uint minLabel = std::min(neighbor_label, vertex);
		      colors[minLabel] = UNKNOWN;		   
		    }
		}
    }
#pragma omp barrier
    unvisited_vertices.clear();
    unvisited_vertices_tail = 0;

    for (int i = 0; i < num_vertices; i++) {
      if (colors[i] == UNKNOWN) {
	unvisited_vertices[unvisited_vertices_tail++] = i;
      }
    }
}  

void GraphColoring::forbidColors() {
	// Search for pair of vertices (u, v) such that exactly one of the vertices is colored. Then, forbid
	// the color of the one having the color on the other
	// version 0.01 - linear search for such vertices
#pragma omp parallel for collapse(1)
  for (unsigned int vertex_ind = 0; vertex_ind < unvisited_vertices_tail; vertex_ind++) {
    const uint vertex = unvisited_vertices[vertex_ind];
    const uint vertex_color = colors[vertex];
    for (uint neighbor_ind = row_ptr[vertex]; neighbor_ind < row_ptr[vertex + 1]; neighbor_ind++) {
      const uint neighbor_label = col_ind[neighbor_ind];
      if (colors[neighbor_label] != UNKNOWN) {
	forbiddens[vertex].insert(colors[neighbor_label]);
      }		   
    }
  }		
}

unsigned int GraphColoring::getNextColor(const uint & vertex) const {
  uint minColor = 0;
  bool colorValid = false;
  while (!colorValid) {
    minColor++;
    colorValid = true;
    for (int neighbor_ind = row_ptr[vertex]; neighbor_ind < row_ptr[vertex + 1] && colorValid; neighbor_ind++) {
      const uint neighbor = col_ind[neighbor_ind];
      if (colors[neighbor] == minColor) {
	colorValid = false;
      }
    }
  }
  return minColor;
}


/*
unsigned int GraphColoring::getNextColor(const uint & vertex) const{
  static vector<bool> neighborColors(100000, false);
  
  for (uint neighbor_ind = row_ptr[vertex]; neighbor_ind < row_ptr[vertex + 1]; neighbor_ind++){
    const uint neighbor = col_ind[neighbor_ind];
    if (colors[neighbor] < neighborColors.size() - 1) {
      neighborColors[colors[neighbor]] = true;
    }
    else {
      neighborColors.resize(neighborColors.size()*2, false);
      neighborColors[colors[neighbor]] = true;
    }
  }

  uint returningColor = UNKNOWN;
  for (uint i = 1; i < neighborColors.size(); i++) {
    if (!neighborColors[i]) {
      returningColor = i;
      break;
    }
  }

  if (returningColor == UNKNOWN) {
    returningColor = neighborColors.size();
  }

  for (uint i = 0; i < neighborColors.size(); i++) {
    neighborColors[i] = false;
  }
  return returningColor;
}
*/
