from collections import defaultdict
from infomap     import Infomap
from typing      import List, Optional

from .codebook   import CodeBook
from .io.reader  import *
from .util       import inits, splitQuotationAware

class PathCost:
    """
    PathCost derives codebooks from the modular structure of a network and uses
    them to caluclate the cost in bits to
    """
    def __init__(self) -> None:
        """
        Initialise.
        """

    def load_from_files(self, netfile: str, treefile: str) -> None:
        """
        Load a partition from files and construct codebooks from it.

        Parameters
        ----------
        netfile: str
            The file that contains the network.

        treefile: str
            The file that contains the partition.
        """
        self._load_modules(netfile, treefile)
        self._process_inputs()
        return self

    def load_from_infomap(self, infomap: Infomap) -> None:
        """
        Load a partition from an Infomap instance and construct codebooks from it.
        """
        raise Exception("PathCost:load_from_infomap not implemented.")

    def _process_inputs(self):
        """
        Private method for populating the codebooks after loading data.
        """

        # create the codebook and insert all paths
        self.cb = CodeBook()
        self.cb.insert_path((), 1, 0, 0)
        for m in self.modules:
            self.cb.insert_path( m.split(":")
                               , self.modules[m]["flow"]
                               , self.modules[m]["enter"]
                               , self.modules[m]["exit"]
                               )
        self.cb.calculate_normalisers()
        self.cb.calculate_costs()

    def _load_modules(self, netfile: str, treefile: str) -> None:
        """
        Load the modules.

        Parameters
        ----------
        netfile: str
            The file that contains the network.

        treefile: str
            The file that contains the partition.
        """
        # for reading the network
        reader  = StateFileReader(netfile)

        # all modules with their nodes
        modules = defaultdict(lambda: dict( nodes = set()
                                          , flow  = 0.0
                                          , enter = 0.0
                                          , exit  = 0.0
                                          )
                             )

        # node visit rates
        flows   = dict()

        # full paths of nodes
        paths   = dict()

        with open(treefile, "r") as fh:
            for line in fh:
                if line.startswith("*Links undirected"):
                    break

                if not line.startswith("#"):
                    path, flow, _, nodeID, _ = splitQuotationAware(line.strip())
                    nodeID = int(nodeID)
                    flows[nodeID] = float(flow)
                    paths[nodeID] = path.split(":")
                    for prefix in inits(paths[nodeID]):
                        module = ":".join(prefix)
                        modules[module]["nodes"].add(nodeID)

        # set the flow of modules by summing over the flow of contained nodes
        for n in reader.get_state_nodes():
            for i,_ in enumerate(paths[n], start=1):
                path = ":".join(paths[n][:i])
                modules[path]["flow"] += flows[n]

        # count all degrees for flow normalisation, assuming that the network is undirected
        degree = defaultdict(lambda: 0)
        for e in reader.get_edges():
            u,v,w = e
            degree[u] += w
            degree[v] += w

        # set enter and exit flow by going over all edges
        for e in reader.get_edges():
            u,v,w = e
            pfrom = paths[u]
            pto   = paths[v]

            i = 0
            while i < len(pfrom) and i < len(pto) and pfrom[i] == pto[i]:
                i += 1
            prefix = ":".join(pfrom[:i])

            pfrom = pfrom[i:]
            pto   = pto[i:]

            for j,_ in enumerate(pfrom, start=1):
                path = ":".join(pfrom[:j])
                if len(prefix) > 0:
                    path = prefix + ":" + path
                modules[path]["exit"]  += w/degree[u] * flows[u]
                modules[path]["enter"] += w/degree[v] * flows[v]

            for j,_ in enumerate(pto, start=1):
                path = ":".join(pto[:j])
                if len(prefix) > 0:
                    path = prefix + ":" + path
                modules[path]["exit"]  += w/degree[v] * flows[v]
                modules[path]["enter"] += w/degree[u] * flows[u]

        self.modules = modules
        self.paths   = paths

    def get_path_cost_directed(self, u: int, v: int) -> float:
        """
        Calculate the path cost for the directed edge (u,v).

        Parameters
        ----------
        u: int
            The source node.

        v: int
            The target node.
        """
        if u == v:
            raise Exception("Those are the same nodes, don't do this.")

        return self.cb.get_path_cost_forward(self.paths[u]) \
             + self.cb.get_walk_cost(self.paths[u], self.paths[v])

    def get_path_cost_undirected(self, u, v):
        """
        Calculate the path cost for the undirected edge {u,v} as the average
        of the path costs of the directed edges (u,v) and (v,u).

        Parameters
        ----------
        u: int
            The source node.

        v: int
            The target node.
        """
        return 0.5 * ( self.get_path_cost_directed(u, v)
                     + self.get_path_cost_directed(v, u)
                     )
