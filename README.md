# Map Equation Similarity (MapSim)
This repository implements Map Equation Similarity (MapSim) in Python: an information-theoretic community-based measure for calculating node similarities.
MapSim is explained in [this paper](https://proceedings.mlr.press/v198/blocker22a.html) where it was used for link prediction.

This repository also implements flow divergence, an information-theoretic divergence measure for comparing network, explained in [this paper](https://arxiv.org/abs/2401.09052).


## Setup
To get started, create a virtual environment, activate it, and install MapSim.

```bash
# crate the virtual environment
virtualenv mapsim-venv

# activate the virtual environment
source mapsim-venv/bin/activate

# install mapsim with pip
pip install git+https://github.com/mapequation/map-equation-similarity
```


## Usage
MapSim uses Infomap to detect communities, but you can also use other/your own communities.

Let's take the following network:

<img src="img/example-network-with-node-ids.svg" alt="An example network with fifteen nodes." width="250"/>

We can define it directly in python with `networkx`, like so:

```python
import networkx as nx
G = nx.Graph()

G.add_edge( 1,  2, weight = 1)
G.add_edge( 1,  3, weight = 1)
G.add_edge( 1,  4, weight = 1)
G.add_edge( 2,  3, weight = 1)
G.add_edge( 3,  4, weight = 1)

G.add_edge( 5,  6, weight = 1)
G.add_edge( 5,  9, weight = 1)
G.add_edge( 6,  7, weight = 1)
G.add_edge( 6,  9, weight = 1)
G.add_edge( 7,  8, weight = 1)
G.add_edge( 7,  9, weight = 1)
G.add_edge( 8,  9, weight = 1)

G.add_edge(10, 11, weight = 1)
G.add_edge(10, 12, weight = 1)
G.add_edge(10, 14, weight = 1)
G.add_edge(10, 15, weight = 1)
G.add_edge(11, 12, weight = 1)
G.add_edge(12, 13, weight = 1)
G.add_edge(12, 14, weight = 1)
G.add_edge(13, 14, weight = 1)
G.add_edge(14, 15, weight = 1)

G.add_edge( 3,  5, weight = 1)
G.add_edge( 4, 11, weight = 1)
G.add_edge( 9, 12, weight = 1)
```

Then, we use `Infomap` do detect communities:
```python
from infomap import Infomap

im = Infomap(silent = True)
im.add_networkx_graph(G)
im.run()
```

Infomap detects three communities, shown with colours.

<img src="img/example-network-with-node-ids-and-communities.svg" alt="An example network with fifteen nodes and three communities." width="250"/>

Next, we create a `MapSim` instance:
```python
from mapsim import MapSim

ms = MapSim().from_infomap(im)
```

Now we can use `MapSim` to calculate similarities between nodes:
```python
> ms.mapsim(3, 1)
0.21428571428571427
> ms.mapsim(3, 5)
0.007936507936507936
```

We can also ask for similarities between nodes that are *not* connected by a link:
```python
> ms.mapsim(3, 12)
0.01082251082251082
```

Generally speaking, nodes are more similar to each other when they are in the same community.

To compare entire partitions, we can use flow divergence.
Consider the following example network with four different partitions:

<img src="img/flow-divergence-motivation.svg" alt="Four different partitions for the same network." width="800"/>

Let's define the network using `networkx` and those partitions as dictionaries where keys are node IDs and values are module assignments,
```python
G_three_triangles = nx.Graph()

# the triangles
G_three_triangles.add_edge(1, 2, weight = 1)
G_three_triangles.add_edge(2, 3, weight = 1)
G_three_triangles.add_edge(3, 1, weight = 1)

G_three_triangles.add_edge(4, 5, weight = 1)
G_three_triangles.add_edge(5, 6, weight = 1)
G_three_triangles.add_edge(6, 4, weight = 1)

G_three_triangles.add_edge(7, 8, weight = 1)
G_three_triangles.add_edge(8, 9, weight = 1)
G_three_triangles.add_edge(9, 7, weight = 1)

# connect the triangles
G_three_triangles.add_edge(2, 4, weight = 1)
G_three_triangles.add_edge(6, 8, weight = 1)
G_three_triangles.add_edge(7, 3, weight = 1)

# partitions
A = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3}
B = {1:1, 2:1, 3:3, 4:1, 5:2, 6:2, 7:3, 8:2, 9:3}
C = {1:1, 2:2, 3:1, 4:2, 5:2, 6:3, 7:1, 8:3, 9:3}
D = {1:3, 2:1, 3:1, 4:2, 5:1, 6:2, 7:3, 8:3, 9:2}
```

And let's load them into `MapSim`
```python
def mapsim_from_modules(G, M):
    im = Infomap(silent = True, no_infomap = True)
    im.add_networkx_graph(G)
    im.run(initial_partition = M)
    return MapSim().from_infomap(im)

ms_A = mapsim_from_modules(G_three_triangles, A)
ms_B = mapsim_from_modules(G_three_triangles, B)
ms_C = mapsim_from_modules(G_three_triangles, C)
ms_D = mapsim_from_modules(G_three_triangles, D)
```

Now we can compare the partitions with flow divergence.
```python
> ms_A.D(ms_A)
0.0
> ms_A.D(ms_B)
1.91508642047648
> ms_A.D(ms_C)
1.91508642047648
> ms_A.D(ms_D)
1.50296642395886
```


## How MapSim works
To explain how MapSim works, let's first take a look at how the map equation works.

Let's imagine a random walker who moves on the network and we want to describe where the random walker is after every step.
We can do this by assigning codewords to each node using a [Huffman code](https://en.wikipedia.org/wiki/Huffman_coding) based on the nodes' visit rates.
The general idea is to use shorter codewords for nodes that the random walker visits often and longer codewords for nodes that the random walker visits less often.
Why?
Because we want to describe the random walk efficiently, which means that the codewords should be as short as possible.
A Huffman code does exactly this.

<img src="img/map-equation-principle-one-level.svg" alt="The nodes' codewords are based on a Huffman code and their visit rates." width="300"/>

The black trace on the network is a part of a possible random walk and the codewords at the bottom show we would encode it.

But it turns out that we can do better if the network has communities:
We build one Huffman code per community.
Then we can use the same codeword for different nodes in different communities.
Overall, this makes the codewords shorter.
But this has a cost: we also need to say when we leave and enter communities.

<img src="img/map-equation-principle-two-level.svg" alt="An example network with fifteen nodes and three communities." width="300"/>

We can see that we need fewer bits when we use the codewords that are based on communities.

MapSim uses the same principles as the map equation, but only thinks about one-step walks.
We draw the coding scheme from above in a tree.

<img src="img/map-equation-principle-coding.svg" alt="An example network with fifteen nodes and three communities." width="300"/>

Now we can ask: how would we encode a step from node 3 to node 1?
We would use the codeword `01`, which costs 2 bits.
To describe the step from node 3 to node 5, we would use the codeword sequence `101` to exit the blue modules, `11` to enter the orange module, and `00` to visit node 5, which requires 7 bits in total.
And to describe the step from node 3 to node 12 - a link that doesn't exist in the network - we would use the codewords `101` to exit the blue module, `0` to enter the green module, and `10` to visit node 12, that is 6 bits in total.

There's only one more detail:
In practice, we don't simulate random walks or assign actual codewords.
Instead, we use node visit rates and transition rates between nodes.
We draw the same coding scheme from aboce again, but this time, we use transition rates.

<img src="img/map-equation-principle-coding-rates.svg" alt="An example network with fifteen nodes and three communities." width="500"/>

How do we read this?
A random walker who is in the blue module visits node 1 with probability $\frac{3}{14}$ and exits module 1 with probability $\frac{2}{14}$.
This means that we get new values for the transitions we considered before:
  * $3 \to 1: \frac{3}{14} \approx 0.214$
  * $3 \to 5: \frac{2}{14} \cdot \frac{2}{6} \cdot \frac{3}{18} \approx 0.008$
  * $3 \to 12: \frac{2}{14} \cdot \frac{2}{6} \cdot \frac{5}{22} \approx 0.011$

These transition rates are the same as above in the "Usage" section.
To "convert" them into costs in bits, we can take their negative $\log_2$:
  * $3 \to 1: -\log_2 \left( \frac{3}{14} \right) \approx 2.22$ bits
  * $3 \to 5: -\log_2 \left( \frac{2}{14} \cdot \frac{2}{6} \cdot \frac{3}{18} \right) \approx 6.98$ bits
  * $3 \to 12: -\log_2 \left( \frac{2}{14} \cdot \frac{2}{6} \cdot \frac{5}{22} \right) \approx 6.53$ bits


## How flow divergence works
Flow divergence combines ideas from the map equation and the Kullback-Leibler divergence.
For two probability distributions $P$ and $Q$ that are defined on the same sample space $X$, the Kullback-Leibler divergence quantifies the expected additional number of bits required to describe samples from $X$ using an estimate $Q$ of its true frequencies $P$,
```math
D_{KL}(P\,||\,Q) = \sum_{x \in X} p_x \log_2 \frac{p_x}{q_x}.
```

Following the same idea, flow divergence quantifies the expected additional number of bits per step for describing a random walk using an estimate B of the networks "true" community structure A.
```math
D_F(\mathsf{A} \,||\, \mathsf{B}) = \sum_{u \in V} p_u \sum_{v \in V} t_{uv}^\mathsf{A} \log_2 \frac{\text{mapsim}(\mathsf{A}, u, v)}{\text{mapsim}(\mathsf{B}, u, v)},
```
where $V$ is the set of nodes in the network, $\mathsf{A}$ and $\mathsf{B}$ are the network partitions we want to compare, and $t_{uv}^\mathsf{A}$ is the transition probability from node $u$ to $v$ based on the reference partition $\mathsf{A}$.


## Citation
If you're using MapSim, please cite
```bibtex
@InProceedings{pmlr-v198-blocker22a,
  title     = {Similarity-Based Link Prediction From Modular Compression of Network Flows},
  author    = {Bl{\"o}cker, Christopher and Smiljani{\'c}, Jelena and Scholtes, Ingo and Rosvall, Martin},
  booktitle = {Proceedings of the First Learning on Graphs Conference},
  pages     = {52:1--52:18},
  year      = {2022},
  editor    = {Rieck, Bastian and Pascanu, Razvan},
  volume    = {198},
  series    = {Proceedings of Machine Learning Research},
  month     = {09--12 Dec},
  publisher = {PMLR},
  pdf       = {https://proceedings.mlr.press/v198/blocker22a/blocker22a.pdf},
  url       = {https://proceedings.mlr.press/v198/blocker22a.html},
}
```

And if you're using flow divergence, please cite
```bibtex
@misc{blöcker2024flow,
  title         = {Flow Divergence: Comparing Maps of Flows with Relative Entropy}, 
  author        = {Christopher Blöcker and Ingo Scholtes},
  year          = {2024},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2401.09052},
  url           = {https://doi.org/10.48550/arXiv.2401.09052},
  eprint        = {2401.09052},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  howpublished  = {\href{https://doi.org/10.48550/arXiv.2401.09052}{arXiv:2401.09052}}
}
```