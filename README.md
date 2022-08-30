# map-equation-similarity
This repository contains a Python implementation of Map Equation Similarity, or MapSim for short, a community-based approach to calculate node similarities.
These node similarities can be used for, for example, link prediction.

## Setup
To get started, create a virtual environment `virtualenv map-equation-similarity-venv` and activate it `source map-equation-similarity-venv/bin/activate`, Then, install MapSim in the virtual environment with `pip install`.

## Usage
MapSim uses Infomap to detect communities.
You can define your network in a file, for example `example.net`, like so
```
# Coding example
*Vertices 9
1 "1"
2 "2"
3 "3"
4 "4"
5 "5"
6 "6"
7 "7"
8 "8"
9 "9"
*Links 10
1 3 1
1 5 1
2 3 1
3 4 1
4 5 1
5 6 1
6 7 1
6 8 1
7 8 1
8 9 1
```

and load it with MapSim when creating an instance.
```
from mapsim import MapSim

ms = MapSim()
ms.run_infomap( netfile                   = "example.net"
              , directed                  = False
              , teleportation_probability = 0
              )
```

Then, you can use MapSim to calculate similarities between nodes (MapSim uses strings as node labels as defined in the input file):
```
ms.get_path_cost_directed("1", "2")
```