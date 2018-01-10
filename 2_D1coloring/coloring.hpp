#ifndef _COLORING_HPP
#define _COLORING_HPP

#include <vector>
#include <iostream>
#include <set>

#ifdef __cplusplus
extern"C" {
#endif

#include "graph.h"

#ifdef __cplusplus
}
#endif

typedef unsigned int uint;

class GraphColoring {
public:
	GraphColoring(etype * row_ptr, vtype* col_ind, int nov);
	~GraphColoring();

	void perform_coloring();
	void printColors() const;
	bool accuracy() const;
  bool allColored() const;
private:
	// 1) Private member variables
	unsigned int num_vertices;
	unsigned int num_colored;
  std::vector<uint> unvisited_vertices;
  uint unvisited_vertices_tail;
	// CSR
	etype * row_ptr;
	vtype * col_ind;

	// For each vertex we keep color and forbidden colors
	unsigned int * colors;
	std::vector<std::set<uint>> forbiddens; // keep a set of forbidden colors for every vertex

	// 2) Private sub-algorithms
	void assignColors();
	void detectConflicts();
	void forbidColors();

	// 3) Helpers
  unsigned int getNextColor(const uint & vertex) const;
};

#endif
