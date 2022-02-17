from src.pathcost.io.reader import PartitionFromInfomap
from src.pathcost import PathCost
from pytest       import approx
from infomap      import Infomap

import networkx as nx

def test_two_triangles_recommendation():
    pc = PathCost()
    pc.run_infomap( netfile                   = "examples/two-triangles.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )
    
    recommendations_1 = pc.make_recommendations("1")
    assert(list(recommendations_1) in [ ["3", "1", "2", "4", "5", "6"]
                                      , ["3", "1", "2", "4", "6", "5"]
                                      , ["3", "2", "1", "4", "5", "6"]
                                      , ["3", "2", "1", "4", "6", "5"]
                                      ])

    recommendations_5 = pc.make_recommendations("5")
    assert(next(recommendations_5) == "4")


def test_coding_example_recommendation():
    pc = PathCost()
    pc.run_infomap( netfile                   = "examples/coding-example.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )
    
    recommendations_1 = pc.make_recommendations("1")
    assert(list(recommendations_1) in [ ["3", "5", "1", "4", "2", "6", "8", "7", "9"]
                                      , ["3", "5", "1", "4", "2", "8", "6", "7", "9"]
                                      , ["3", "5", "4", "1", "2", "6", "8", "7", "9"]
                                      , ["3", "5", "4", "1", "2", "8", "6", "7", "9"]
                                      ])


def test_random_graph():
    G = nx.generators.erdos_renyi_graph(n = 10, p = 0.5, seed = 0)

    im      = Infomap(silent = True)
    mapping = im.add_networkx_graph(G)
    im.run()

    pc     = PathCost().from_infomap(im, mapping = mapping)
    from_1 = [ n for (n,_) in sorted(pc.predict_interaction_rates(1).items(), key = lambda p: p[1], reverse = True) ]

    assert(list(pc.make_recommendations(1)) == from_1)