from src.mapsim import MapSim
from pytest     import approx

def test_example_network():
    ms = MapSim()
    ms.run_infomap( netfile                   = "examples/coding-example.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )

    probs = ms.predict_interaction_probabilities("1")

    expect = { "1" : 2/12
             , "2" : 1/12
             , "3" : 3/12
             , "4" : 2/12
             , "5" : 3/12
             , "6" : 1/12 * 1/2 * 3/10
             , "7" : 1/12 * 1/2 * 2/10
             , "8" : 1/12 * 1/2 * 3/10
             , "9" : 1/12 * 1/2 * 1/10
             }

    s      = sum(expect.values())
    expect = { k : v/s for (k,v) in expect.items() }

    assert approx(probs["1"], 0.01) == expect["1"]
