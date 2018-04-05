# CS406 Parallel Computing
<h4>Homework &amp; Project of CS406/531 Parallel Computing</h4>
This repo contains the homework and our group project for the Parallel Computation course. A detailed report for each homework is provided within their respective folders. A brief overview of the contents of this repo is provided below.
<hr />
<h2>Homework 1 - Parallel BFS</h2>
<b>Task: </b> Implement BFS parallelized on CPU with OpenMP
<br/>
<b>Solution: </b>
Implementation uses the idea of <a href="https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf">direction-optimized</a> BFS. After each frontier expansion, the program decides which approach (top-down or bottom-up) would perform faster for the next round and proceeds until BFS is finished.

<h2>Homework 2 - Distance-1 Graph Coloring</h2>
<b>Task: </b> Impelement an algorithm, providing accurate distance-1 coloring of the supplied graph that runs on CPU in parallel with OpenMP.
<br/>
<b>Solution: </b>
Moving with the ideas introduced by Deveci et al., the implementation I provide performs three step iterations until each vertex is colored. Each iteration has three steps:
<ol>
  <li>Assign Colors</li>
  <li>Detect Conflicts</li>
  <li>Forbid Colors</li>
</ol>

<h2>Homework 3 - Shortness Centrality</h2>
<b>Task: </b> Implement a GPU parallelized algorithm to compute shortness centrality of a provided graph.
<br/>
<b>Solution: </b>
The implementation assigns each CUDA core a vertex. Each thread is responsible for performing BFS having its assigned vertex as source node. However, memory requirement of my implementation is very high; therefore, the implementation works only for small graphs.

<h2>Homework 4 - Nearest Neighbor Search</h2>
<b>Task: </b> Provided a set of training and testing points on a 16-dimensional space, find the nearest neighbor among training points of each testing point.
<br/>
<b>Solution: </b>
The implementation assigns each CUDA core a testing point. Afterwards, each core iterates over the training points to find the training point with smallest euclidean distance. CUDA streams were used to overlap computation with communication.

<h2>Project - Distance-2 Graph Coloring</h2>
<b>Task: </b> Impelement CPU, GPU and Heterogeneous algorithms solving the distance-2 graph coloring problem.
<br/>
<b>Solution: </b>
We have extended the implementation of distance-1 graph coloring to distance-2 with improvements. OpenMP algorithm uses thread-private forbidden arrays and CUDA implementation uses shared block-cache to store bit vectors as forbidden arrays.
