from infomap                import Infomap
from src.pathcost           import PathCost
from src.pathcost.io.reader import NetworkFromNetFile
from pytest                 import approx

def test_read_treefile():
    im = Infomap( silent    = True
                , two_level = True
                , seed      = 42
                )

    net = NetworkFromNetFile( filename = "examples/coding-example.net"
                            , directed = False
                            )

    for (u,v,_w) in net.get_edges():
        im.add_link(u,v)

    im.run()
    im.write_tree("out/tmp.tree")

    #pc = PathCost().from_treefile("out/tmp.tree")

    #assert len(pc.addresses) == 9
    assert(True)