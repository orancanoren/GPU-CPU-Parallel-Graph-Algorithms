coloring: coloring.cpp
	gcc graphio.c -c -O3
	gcc mmio.c -c -O3
#	gcc graph.c -c -O3	
	g++ coloring.cpp coloring.hpp main.cpp -fopenmp -c -std=c++11 -O3
	gcc -o coloring coloring.o mmio.o graphio.o main.o -fopenmp -lstdc++ -O3
clean:
	rm coloring *.o *.gch *.bin
