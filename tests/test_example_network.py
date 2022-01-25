from src.pathcost import PathCost
from pytest       import approx

def test_example_network():
    pc = PathCost()
    pc.run_infomap( netfile                   = "examples/coding-example.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )

    probs = pc.predict_interaction_probabilities("1")

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
    assert approx(probs["2"], 0.01) == expect["2"]
    assert approx(probs["3"], 0.01) == expect["3"]
    assert approx(probs["4"], 0.01) == expect["4"]
    assert approx(probs["5"], 0.01) == expect["5"]
    assert approx(probs["6"], 0.01) == expect["6"]
    assert approx(probs["7"], 0.01) == expect["7"]
    assert approx(probs["8"], 0.01) == expect["8"]
    assert approx(probs["9"], 0.01) == expect["9"]
