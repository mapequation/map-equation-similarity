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

        partition = PartitionFromTreeFile(treefile)

        self._process_inputs()
        return self

    def from_infomap(self, infomap: Infomap) -> None:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
        """
        partition    = PartitionFromInfomap(infomap)

        self.modules = partition.get_modules()
        self.paths   = partition.get_paths()
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

        infomap = Infomap(" ".join(infomap_args))
        infomap.read_file(netfile)
        infomap.run()

        return self.from_infomap(infomap)

    def _process_inputs(self):
        """
        Private method for populating the codebooks after loading data.
        """
        # create the codebook and insert all paths
        self.cb = CodeBook()
        for m in self.modules:
            self.cb.insert_path( path  = m
                               , flow  = self.modules[m]["flow"]
                               , enter = self.modules[m]["enter"]
                               , exit  = self.modules[m]["exit"]
                               )
        self.cb.calculate_normalisers()
        self.cb.calculate_costs()

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
