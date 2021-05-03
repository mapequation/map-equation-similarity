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

        # check if there are states in the input file
        with open(netfile, "r") as fh:
            has_states = False
            for line in fh:
                has_states = line.startswith("*States")
                if has_states:
                    break

        # if there are states, read it as a state file and work with state nodes
        if has_states:
            network = NetworkFromStateFile(netfile)
        else:
            network = NetworkFromNetFile(netfile)

        partition = PartitionFromTreeFile(treefile)

        self._load_modules(network, partition)
        self._process_inputs()
        return self

    def load_from_infomap_run(self, netfile: str, directed: bool = False) -> None:
        """
        Run infomap on the supplied network file and use the partition it finds.

        Parameters
        ----------
        netfile: str
            The file that contains the network.

        directed: bool = False
            Whether the network is directed or not.
        """

        # run infomap
        infomap_args = ["--silent"]
        if directed:
            infomap_args.append("--directed")

        im = Infomap(" ".join(infomap_args))
        im.read_file(netfile)
        im.run()

        partition    = PartitionFromInfomap(im)
        self.modules = partition.get_modules()
        self.paths   = partition.get_paths()

        # # but we need to look at the input file again
        # if im.memoryInput:
        #     network = NetworkFromStateFile(netfile, directed = directed)
        # else:
        #     network = NetworkFromNetFile(netfile, directed = directed)

        self._process_inputs()
        return self

    def _process_inputs(self):
        """
        Private method for populating the codebooks after loading data.
        """
        # create the codebook and insert all paths
        self.cb = CodeBook()
        self.cb.insert_path((), 1, 0, 0)
        for m in self.modules:
            self.cb.insert_path( path  = m
                               , flow  = self.modules[m]["flow"]
                               , enter = self.modules[m]["enter"]
                               , exit  = self.modules[m]["exit"]
                               )
        self.cb.calculate_normalisers()
        self.cb.calculate_costs()

    def _load_modules(self, network: Network, partition: Partition) -> None:
        """
        Load the modules.

        Parameters
        ----------
        network: Network
            The network.

        partition: Partition
            The partition.
        """
        self.modules = partition.get_modules()
        self.paths   = partition.get_paths()

        # flows = partition.get_flows()
        #
        # # set the flow of modules by summing over the flow of contained nodes
        # for n in network.get_nodes():
        #     for i,_ in enumerate(self.paths[n], start=1):
        #         path = tuple(self.paths[n][:i])
        #         self.modules[path]["flow"] += flows[n]
        #
        # # count all out degrees for flow normalisation
        # degree = defaultdict(lambda: 0)
        # for e in network.get_edges():
        #     u,v,w = e
        #     degree[u] += w
        #     degree[v] += w
        #
        # # set enter and exit flow by going over all edges
        # for e in network.get_edges():
        #     u,v,w = e
        #     pfrom = self.paths[u]
        #     pto   = self.paths[v]
        #
        #     i = 0
        #     while i < len(pfrom) and i < len(pto) and pfrom[i] == pto[i]:
        #         i += 1
        #     prefix = tuple(pfrom[:i])
        #
        #     pfrom = pfrom[i:]
        #     pto   = pto[i:]
        #
        #     for j,_ in enumerate(pfrom, start=1):
        #         path = tuple(pfrom[:j])
        #         if len(prefix) > 0:
        #             path = prefix + path
        #         self.modules[path]["exit"]  += w/degree[u] * flows[u]
        #         self.modules[path]["enter"] += w/degree[v] * flows[v]
        #
        #     for j,_ in enumerate(pto, start=1):
        #         path = tuple(pto[:j])
        #         if len(prefix) > 0:
        #             path = prefix + path
        #         self.modules[path]["exit"]  += w/degree[v] * flows[v]
        #         self.modules[path]["enter"] += w/degree[u] * flows[u]

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
