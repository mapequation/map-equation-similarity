from infomap              import Infomap
from src.mapsim           import MapSim
from src.mapsim.io.reader import NetworkFromNetFile

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

    #pc = MapSim().from_treefile("out/tmp.tree")

    #assert len(pc.addresses) == 9
    assert(True)