#ifndef GRAPH_H
#define GRAPH_H

#define vtype int
#define etype unsigned int
#define ewtype double
#define vwtype int
#define cltype int

#define _UNKNOWN -1

int graphCheck(etype* xadj, vtype* adj, ewtype* ewghts, vwtype* vwghts, vtype nov, int loop);
int order_bfs(etype* xadj, vtype* adj, vtype nov, vtype* que, int* levels);
int transpose(etype* xadj, vtype* adj,  ewtype* ewghts, vtype nov, etype* txadj, vtype* tadj, ewtype* tewghts);
int extract_largest_comp(etype* xadj, vtype* adj, vtype nov, vtype* que, vtype* compid, etype** lxadj, vtype** ladj, vtype* lnov);

#endif






