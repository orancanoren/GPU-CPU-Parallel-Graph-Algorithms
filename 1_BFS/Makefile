bfs: bfs.c
	gcc graphio.c -c -O3
	gcc mmio.c -c -O3
#	gcc graph.c -c -O3	
	gcc bfs.c -fopenmp -c -O3
	gcc -o bfs bfs.o mmio.o graphio.o -g -fopenmp
clean:
	rm bfs *.o *~
