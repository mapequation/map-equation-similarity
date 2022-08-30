from infomap    import Infomap
from src.mapsim import MapSim
from itertools  import islice
from time       import time

import networkx as nx

def test_two_triangles_recommendation():
    ms = MapSim()
    ms.run_infomap( netfile                   = "examples/two-triangles.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )
    
    recommendations_1 = ms.make_recommendations("1")
    assert(list(recommendations_1) in [ ["3", "1", "2", "4", "5", "6"]
                                      , ["3", "1", "2", "4", "6", "5"]
                                      , ["3", "2", "1", "4", "5", "6"]
                                      , ["3", "2", "1", "4", "6", "5"]
                                      ])

    recommendations_5 = ms.make_recommendations("5")
    assert(next(recommendations_5) == "4")


def test_coding_example_recommendation():
    ms = MapSim()
    ms.run_infomap( netfile                   = "examples/coding-example.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )
    
    recommendations_1 = ms.make_recommendations("1")
    assert(list(recommendations_1) in [ ["3", "5", "1", "4", "2", "6", "8", "7", "9"]
                                      , ["3", "5", "1", "4", "2", "8", "6", "7", "9"]
                                      , ["3", "5", "4", "1", "2", "6", "8", "7", "9"]
                                      , ["3", "5", "4", "1", "2", "8", "6", "7", "9"]
                                      ])


# ToDo: nodes with the same similarity can end up listed in arbitrary order and make the test fail.
def test_random_graph():
    G = nx.generators.erdos_renyi_graph(n = 10, p = 0.5, seed = 0)

    im      = Infomap(silent = True)
    mapping = im.add_networkx_graph(G)
    im.run()

    ms     = MapSim().from_infomap(im, mapping = mapping)
    from_1 = [ n for (n,_) in sorted(ms.predict_interaction_rates(1).items(), key = lambda p: p[1], reverse = True) ]

    assert(list(ms.make_recommendations(1)) == from_1)


def test_performance():
    G = nx.generators.connected_caveman_graph(l = 15, k = 10)

    im = Infomap(silent = True)
    mapping = im.add_networkx_graph(G)
    im.run()

    ms = MapSim().from_infomap(im, mapping = mapping)

    start                  = time()
    strict_recommendations = [ n for (n,_) in sorted(ms.predict_interaction_rates(0).items(), key = lambda p: p[1], reverse = True)][:10]
    between                = time()
    lazy_recommendations   = list(islice(ms.make_recommendations(0), 10))
    end                    = time()

    strict_time = between - start
    lazy_time   = end - between

    print(strict_time, lazy_time)

    assert(lazy_time < strict_time)