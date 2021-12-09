from pathcost import PathCost
from pytest   import approx

def test_example_network():
    pc = PathCost()
    pc.run_infomap( netfile                   = "examples/coding-example.net"
                  , directed                  = False
                  , teleportation_probability = 0
                  )

    probs = pc.predict_next_element_rates("1")

    assert approx(probs["1"], 0.01) == 0.10 / 0.6
    assert approx(probs["2"], 0.01) == 0.05 / 0.6
    assert approx(probs["3"], 0.01) == 0.15 / 0.6
    assert approx(probs["4"], 0.01) == 0.10 / 0.6
    assert approx(probs["5"], 0.01) == 0.15 / 0.6
    assert approx(probs["6"], 0.01) == 0.05 / 0.6 * 0.05 / 0.10 * 0.15 / 0.5
    assert approx(probs["7"], 0.01) == 0.05 / 0.6 * 0.05 / 0.10 * 0.10 / 0.5
    assert approx(probs["8"], 0.01) == 0.05 / 0.6 * 0.05 / 0.10 * 0.15 / 0.5
    assert approx(probs["9"], 0.01) == 0.05 / 0.6 * 0.05 / 0.10 * 0.05 / 0.5
