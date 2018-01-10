#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mmio.h"
#include "graphio.h"

//#define DEBUG

typedef struct {
  vtype i;
  vtype j;
  ewtype w;
} Triple;

int cmp(const void *a, const void *b){
	const vtype *ia = (const vtype *)a;
	const vtype *ib = (const vtype *)b;
	return *ia  - *ib;
}

int tricmp(const void *t1, const void *t2){
  Triple *tr1 = (Triple *)t1;
  Triple *tr2 = (Triple *)t2;
  if(tr1->i == tr2->i) {
    return (int)(tr1->j - tr2->j);
  }
  return (int)(tr1->i - tr2->i);
}

int ends_with(const char *str, const char *suffix) {
  if (!str || !suffix) return 0;
  size_t lenstr = strlen(str);
  size_t lensuffix = strlen(suffix);
  if (lensuffix >  lenstr) return 0;
  return (strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0);
}

int read_chaco(FILE* fp, etype **pxadj, vtype **padj,
				       ewtype **pewghts, vwtype **pvwghts,
				       vtype* pnov, int loop) {

	int state = 0, fmt = 0, ncon = 1, i;
	vtype numVertices = -1, vcount = 0, jv;
	etype numEdges = -1, ecount = 0;
	char *temp, *graphLine =  (char *) malloc(sizeof(char)*10000000+1);


	while(fgets(graphLine, 10000000, fp) != NULL) {
		for(i = 0; i < (int) strlen(graphLine); i++) {
			char c = graphLine[i];
			if(c != ' ' && c != '\t' && c != '\n') {
				break;
			}
		}

		/* read the line wrt fmt and mw */
		if(graphLine[0] == '%') {
			continue;
		} else if(state == 0) {
			temp = strtok(graphLine, " \t\n");
			numVertices = atoi(temp);

			temp = strtok(NULL, " \t\n");
			numEdges = atoi(temp);

			temp = strtok(NULL, " \t\n");
			if(temp != NULL) {
				fmt = atoi(temp);
				temp = strtok(NULL, " \t\n");
				if(temp != NULL) {ncon = atoi(temp);}
			}

			*pnov = numVertices;
			(*pxadj) = (etype*)malloc(sizeof(etype) * (numVertices+1));
			(*pxadj)[0] = 0;

			(*pvwghts)  = (vwtype*)malloc(sizeof(vwtype) * numVertices);
			(*padj) = (vtype*)malloc(sizeof(vtype) * 2 * numEdges);
			(*pewghts) = (ewtype*)malloc(sizeof(ewtype)  * 2 * numEdges);

            state = 1;
		} else { /* consume a line */
			if(vcount == numVertices) {
				printf("Error: file contains more than %ld lines\n", (long int)numVertices);
				return -1;
			}

			/* read the line wrt fmt and mw */
			temp = strtok(graphLine, " \t\n");

			/* ignore vertex size if exists */
			if (fmt >= 100) {
				temp = strtok(NULL, " \t\n");
			}

			/* take only the first vertex weight if exists. ignore the rest
			 * if no vertex weight exists then unit weight */
			if (fmt % 100 >= 10) {
				(*pvwghts)[vcount] = atoi(temp);
				for (i = 1; i < ncon; i++) {
					temp = strtok(NULL, " \t\n");
				}
			} else {
				(*pvwghts)[vcount] = 1;
			}

	        /* read edges and edge weights if exists */
			while(temp != NULL) {
				if(ecount == 2 * numEdges) {
					printf("Error: file contains more than %ld edges\n", (long int)numEdges);
					return -1;
				}

				(*padj)[ecount] = atoi(temp) - 1; /* ids start from 1 in the graph */
				if((*padj)[ecount] == vcount && !loop) {
					continue;
				}

				temp = strtok(NULL, " \t\n");
				if(fmt % 10 == 1) {
					(*pewghts)[ecount] = atoi(temp);
					temp = strtok(NULL, " \t\n");
				} else {
					(*pewghts)[ecount] = 1;
				}

				if((*pewghts)[ecount] < 0) {
					printf("negative edge weight %lf between %ld-%ld.\n", (*pewghts)[ecount], (long int)vcount, (long int)((*padj)[ecount]));
					return -1;
				}
	            ecount++;
			}

			vcount++;
			(*pxadj)[vcount] = ecount;
		}
	}

	if(vcount != numVertices) {
		printf("number of vertices do not match %ld %ld\n", (long int)numVertices, (long int)vcount);
		return -1;
	}


	if(ecount != 2 * numEdges) {
		printf("number of edges do not match %ld %ld: realloc memory appropriately\n", (long int)ecount, (long int)(2 * numEdges));
		(*padj) = (vtype*)realloc((*padj), sizeof(vtype) * ecount);
		(*pewghts) = (ewtype*)realloc((*pewghts) , sizeof(ewtype) * ecount);
	}

	for(jv = 0; jv < vcount; jv++) {
		qsort((*padj) + (*pxadj)[jv], (*pxadj)[jv+1] - (*pxadj)[jv], sizeof(vtype), cmp);
	}

	return 1;
}

int read_mtx(FILE* fp, etype **pxadj, vtype **padj,
				       ewtype **pewghts, vwtype **pvwghts,
				       vtype* pnov, int loop, int zerobased) {
  Triple *T;
  vtype i, j, M, N, lower, upper, wi, nz;
  ewtype w;
  etype k, ecnt, cnt;

  MM_typecode matcode;

  if (mm_read_banner(fp, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return -1;
  }

  // find out size of sparse matrix ....
  if (mm_read_mtx_crd_size(fp, &M, &N, &nz) != 0) {
    printf("ret code is wrong\n");
    return -1;
  }

#ifdef DEBUG
  printf("There are %ld rows, %ld columns, %ld nonzeros\n", (long int)M, (long int)N, (long int)nz);
#endif

  if(M != N) {
    printf("not a graph, rectangular matrix???\n");
    return -1;
  }

  lower = 1 - zerobased;
  upper = N - zerobased;

  T = (Triple *)malloc(2 * nz * sizeof(Triple));
  cnt = 0;
  if(mm_is_pattern(matcode)) {
    for (ecnt = 0; ecnt < nz; ecnt++) {
      fscanf(fp, "%d %d\n", &i, &j);

      if(i < lower || j < lower || i > upper || j > upper) {
    	  printf("read coordinate %ld %ld -- lower and upper is %ld and %ld\n", (long int)i, (long int)j, (long int)lower, (long int)upper);
    	  return -1;
      }

      if(loop || i != j) {
		  T[cnt].i = i;
		  T[cnt].j = j;
		  T[cnt].w = 1;
		  cnt++;

		  if(mm_is_symmetric(matcode) && i != j) {
			  T[cnt].i = T[cnt-1].j;  /* insert the symmetric edge */
			  T[cnt].j = T[cnt-1].i;
			  T[cnt].w = 1;
			  cnt++;
		  }
      }
    }
  } else {
    for (ecnt = 0; ecnt < nz; ecnt++) {
      if(mm_is_real(matcode)) {
	fscanf(fp, "%d %d %lf\n", &i, &j, &w);
      } else if(mm_is_integer(matcode)) {
	fscanf(fp, "%d %d %d\n", &i, &j, &wi);
      }
      w = (double)wi;

      if(i < lower || j < lower || i > upper || j > upper) {
    	  printf("read coordinate %ld %ld -- lower and upper is %ld and %ld\n", (long int)i, (long int)j, (long int)lower, (long int)upper);
    	  return -1;
      }

      if(w != 0 && (loop || i != j)) {
		  T[cnt].i = i;
		  T[cnt].j = j;
		  T[cnt].w = fabs(w);
		  cnt++;

		  if(mm_is_symmetric(matcode) && i != j) { /* insert the symmetric edge */
			  T[cnt].i = T[cnt-1].j;
			  T[cnt].j = T[cnt-1].i;
			  T[cnt].w = T[cnt-1].w;
			  cnt++;
		  }
      }
    }
  }
  //#ifdef DEBUG
    // printf("the triplet array is filled with %ld edges\n", (long int)cnt); fflush(0);
  //#endif

  /* eliminate the duplicates, and writes them to the arrays */
  qsort(T, cnt, sizeof(Triple), tricmp);

  /* create xadj array */
  (*pxadj) = (etype*)malloc(sizeof(etype) * (N+1));
  memset((*pxadj), 0, sizeof(etype) * (N+1));

  k = 0;
  (*pxadj)[T[0].i + zerobased]++;
  for(ecnt = 1; ecnt < cnt; ecnt++) {
	  i = T[ecnt].i;
	  if(i != T[ecnt-1].i || T[ecnt].j != T[ecnt-1].j) { /* if this edge is not the same as previous one */
		  (*pxadj)[i + zerobased]++;
		  k = i; /* the first edge entry */
	  } else { /* add the weight to the original */
		  T[k].w += T[i].w; /* add the weight to the representative */
	  }
  }
  for(i = 2; i <= N; i++) (*pxadj)[i] += (*pxadj)[i-1];

  (*padj) = (vtype*)malloc(sizeof(vtype) * (*pxadj)[N]);
  (*pewghts) = (ewtype*)malloc(sizeof(ewtype) * (*pxadj)[N]);
  (*padj)[0] = T[0].j - 1 + zerobased; (*pewghts) [0] = T[0].w; k = 1;

  for(ecnt = 1; ecnt < cnt; ecnt++) {
	 i = T[ecnt].i;
	 if(i != T[ecnt-1].i || T[ecnt].j != T[ecnt-1].j) { /* if this edge is not the same as previous one */
		 (*padj)[k] = T[ecnt].j - 1 + zerobased; /* adjust from 1-based to 0-based */
		 (*pewghts)[k++] = T[ecnt].w;
	 }
  }

  (*pvwghts)  = (vwtype*)malloc(sizeof(vwtype) * N);
  for(i = 0; i < N; i++) (*pvwghts) [i] = 1;

  *pnov = N; // the number of vertices
  //#ifdef DEBUG
  //printf("file is read m %ld n %ld edges %ld =? %ld, real edges\n", (long int)M, (long int)N, (long int)k, (long int)((*pxadj)[N]));
  //#endif

  free(T);
  return 1;
}

int readBinaryGraph(FILE* bp, etype **pxadj, vtype **padj,
				       ewtype **pewghts, vwtype **pvwghts,
				       vtype* pnov) {

	fread(pnov, sizeof(vtype), 1, bp);

	(*pxadj) = (etype*)malloc(sizeof(etype) * (*pnov + 1));
	fread(*pxadj, sizeof(etype), (size_t)(*pnov + 1), bp);

	(*padj) = (vtype*)malloc(sizeof(vtype) * (*pxadj)[*pnov]);
	fread(*padj, sizeof(vtype), (size_t)(*pxadj)[*pnov], bp);

	(*pewghts) = (ewtype*)malloc(sizeof(ewtype) * (*pxadj)[*pnov]);
	fread(*pewghts, sizeof(ewtype), (size_t)(*pxadj)[*pnov], bp);

	(*pvwghts)  = (vwtype*)malloc(sizeof(vwtype) * (*pnov));
	fread(*pvwghts, sizeof(vwtype), *pnov, bp);

	return 1;
}

int writeBinaryGraph(FILE* bp, etype *xadj, vtype *adj,
				       ewtype *ewghts, vwtype *vwghts,
				       vtype nov) {

	fwrite(&nov, sizeof(vtype), (size_t)1, bp);
	fwrite(xadj, sizeof(etype), (size_t)(nov + 1), bp);
	fwrite(adj, sizeof(vtype), (size_t)(xadj[nov]), bp);
	fwrite(ewghts, sizeof(ewtype), (size_t)(xadj[nov]), bp);
	fwrite(vwghts, sizeof(vwtype), (size_t)(nov), bp);

	return 1;
}

int read_graph(char* gfile, etype **xadj, vtype **adj,
							ewtype **ewghts, vwtype **vwghts,
							vtype* nov, int loop) {
  char bfile[2048];
  FILE *bp, *fp;

  /*
    Write the binaryfile in the directory of the executable.
  */

  int i=0;
  int dirindex = 0;

  while(gfile[i] != '\0' && i < 2048 ){
    if(gfile[i] == '/')
      {
	dirindex = i;
      }
    i++;
  }
  if(i==2048)
    i = 0;

  char tbfile[2048];
  tbfile[0] = '.';
  char c = ' ';
  i = 0;
  while(c!= '\0'){
    c = gfile[dirindex+i];
    tbfile[i+1] = c;
    i++;
  }



  /* check if binary file exists */
  sprintf(bfile, "%s.bin", tbfile);
  //printf("bin:%s\n", bfile);
  bp = fopen(bfile, "rb");
  if(bp != NULL) { /* read from binary */
	  if(readBinaryGraph(bp, xadj, adj, ewghts, vwghts, nov) == -1) {
		  printf("error reading the graph in binary format\n");
		  fclose(bp);
		  return -1;
	  }
	  fclose(bp);
  } else { /* read from text, write to binary */
    fp = fopen(gfile, "r");
    if(fp == NULL) {
      printf("%s: file does not exist\n", gfile);
      return -1;
    } else {
      if(ends_with(gfile, ".mtx")) {
    	  if(read_mtx(fp, xadj, adj, ewghts, vwghts, nov, loop, 0) == -1) {
    		  printf("problem with mtx file\n");
    		  fclose(fp);
    		  return -1;
    	  }
      } else if(ends_with(gfile, ".txt")) {
    	  if(read_mtx(fp, xadj, adj, ewghts, vwghts, nov, loop, 1) == -1) {
    		  printf("problem with txt file\n");
    		  fclose(fp);
    		  return -1;
    	  }
      } else if(ends_with(gfile, ".graph")) {
    	  if(read_chaco(fp, xadj, adj, ewghts, vwghts, nov, loop) == -1) {
    		  printf("problem with mtx file\n");
    		  fclose(fp);
    	      return -1;
    	  }
      }
      bp = fopen(bfile, "wb");
      if(bp != NULL) {
		  if(writeBinaryGraph(bp, *xadj, *adj, *ewghts, *vwghts, *nov) == -1) {
			  printf("error writing to graph in binary format\n");
			  fclose(bp);
			  return -1;
		  }
		  fclose(bp);
      }

      fclose(fp);
    }
  }
  return 1;
}
