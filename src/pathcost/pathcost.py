from __future__ import annotations

from infomap      import Infomap
from numpy        import log2
from numpy.random import choice
from typing       import List, Optional as Maybe, Tuple

from .codebook    import CodeBook
from .io.reader   import *

class PathCost():
    """
    PathCost derives codebooks from the modular structure of a network and uses
    them to caluclate the cost in bits to
    """
    def __init__(self) -> None:
        """
        Initialise.
        """

    def from_infomap(self, infomap: Infomap, netfile : Maybe[str] = None) -> PathCost:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.

        netfile: Maybe[str] = none
            The file that contains the network (needed for of memory networks).
        """
        partition      : PartitionFromInfomap                                     = PartitionFromInfomap(infomap)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : DefaultDict[str, Dict[str, Tuple[int, ...]]]             = partition.get_paths()
        self._build_codebooks()
        return self

    def run_infomap( self
                   , netfile                   : str
                   , directed                  : bool
                   , teleportation_probability : float
                   , trials                    : int  = 1
                   , seed                      : int  = 42
                   , one_level                 : bool = False
                   , rawdir                    : bool = False
                   ) -> PathCost:
        """
        Run infomap on the supplied network file and use the partition it finds.

        Parameters
        ----------
        netfile : str
            The file that contains the network.

        directed : bool = False
            Whether the network is directed or not.

        teleportation_probability : float
            The teleportation probability.

        trials : int = 1
            Number of trials that infomap should run.

        seed : int = 42
            The seed for infomap.

        one_level : bool = False
            Controls whether to run infomap search or simply output the one-level partition.

        rawdir : bool = False
            Whether to use `-f rawdir` option.
        """

        # run infomap
        infomap_args = [f"--silent --num-trials {trials} --seed {seed} --teleportation-probability {teleportation_probability}"]

        if directed:
            infomap_args.append("--directed")

        if one_level:
            infomap_args.append("--no-infomap")

        if rawdir:
            infomap_args.append("-f rawdir")

        self.infomap = Infomap(" ".join(infomap_args))
        self.infomap.read_file(netfile)
        self.infomap.run()

        self.memory : bool = self.infomap.memoryInput

        # extract the partition from infomap
        partition      = PartitionFromInfomap(self.infomap, netfile)
        self.modules   = partition.get_modules()
        self.addresses = partition.get_paths()
        self._build_codebooks()

        # for making predictions later
        self.node_IDs_to_labels : Dict[int, str] = self.infomap.names

        return self

    def _build_codebooks(self) -> None:
        """
        Private method for populating the codebooks after loading data.
        """
        # create the codebook and insert all paths
        self.cb : CodeBook = CodeBook()
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
        return -log2(self.cb.get_walk_rate(self.addresses[u], self.addresses[v]))

    def get_path_cost_undirected(self, u: int, v: int) -> float:
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
        return -0.5 * ( log2(self.get_path_cost_directed(u, v))
                      + log2(self.get_path_cost_directed(v, u))
                      )

    def get_address(self, path: Tuple[str, ...]) -> Tuple[int, ...]:
        """
        """
        # if we don't use memory, we forget all history and simply
        # get the address of the last node in the path
        if not self.memory:
            return self.addresses[path[-1]]

        # but if there is memory, we have to select the address in
        # the correct node that respects the given path
        else:
            # if the path is empty, we start at the root of the partition
            if len(path) == 0:
                return ()

            # if the path only contains one node, we start at the epsilon
            # node of the respective physical node
            if len(path) == 1:
                return (self.addresses[path[0]]["{}"],)

            # if the path contains two nodes, we start at the corresponding
            # memory node if it has an own address, otherwise we begin at the
            # epsilon node of the physical node, which has the same address
            # as all other memory nodes in that physical node that don't have
            # an own address
            if path[-2] in self.addresses[path[-1]]:
                return self.addresses[path[-1]][path[-2]]
            else:
                return self.addresses[path[-1]]["{}"]


    def predict_next_element(self, path: Tuple[str, ...]) -> str:
        """
        Predicts the next element, given a list of labels of those nodes that
        have interacted.

        Parameters
        ----------
        path: List[str]]
            A list of labels of those nodes that have interacted.

        Returns
        -------
        int
            The label of the most likely next node.
        """
        # ToDo: make a more efficient implementation
        return self.rank_next_elements(path = path)[0]


    def rank_next_elements(self, path: Tuple[str, ...]) -> List[str]:
        """
        Returns a list of node IDs, ranked by
        """
        if len(path) == 0:
            raise Exception("Ranking with empty path not implemented.")

        source_address = self.get_address(path)

        costs : List[Tuple[str, float]] = list()
        for node_label, addresses in self.addresses.items():
            # do not calculate paths from one node to itself
            if node_label != path[-1]:
                if not self.memory:
                    best_cost = self.cb.get_walk_cost(source = source_address, target = addresses)
                else:
                    best_cost = min([ self.cb.get_walk_cost(source = source_address, target = target_address)
                                          for (_memory, target_address) in addresses.items()
                                    ])
                costs.append((node_label, best_cost))

        ranking, _costs = zip(*sorted(costs, key = lambda pair: pair[1]))

        return ranking


    def predict_next_element_probabilities( self
                                          , path: Tuple[str, ...]
                                          , include_self_links : bool = True
                                          ) -> Dict[str, float]:
        """
        Returns a dictionary with node labels as keys and the probabilities that
        the respective node is the next node as values.

        Parameters
        ----------
        path: Tuple[str, ...]
            The path to the start node.

        include_self_links: bool = True
            Whether to include self-links.

        Returns
        -------
        Dict[str, float]
            A dictionary with node labels as keys and the probabilities that
            the respective node is the next node as values.
        """
        if len(path) == 0:
            raise Exception("Predictions with empty path not implemented.")

        source_address = self.get_address(path)

        if not self.memory:
            rates = { node_label : self.cb.get_walk_rate(source = source_address, target = target_address)
                          for node_label, target_address in self.addresses.items()
                          if include_self_links or target_address != source_address
                    }

        else:
            rates = { node_label : sum([ self.cb.get_walk_rate(source = source_address, target = target_address)
                                             for target_address in addresses.values()
                                       ])
                          for node_label, addresses in self.addresses.items()
                          if include_self_links or not source_address in addresses
                    }

        s = sum(rates.values())
        rates = { k : v /s for (k,v) in rates.items() }
        return rates

    def generate_network( self
                        , num_links : int
                        , include_self_links : bool = True
                        ) -> Dict[Tuple[str, str], int]:
        res : Dict[Tuple[str, str], int] = dict()

        nodes_and_flows = [ (node_label, self.cb.get_flow(self.get_address(node_label)))
                                for node_label in self.node_IDs_to_labels.values()
                          ]
        nodes, flows = zip(*nodes_and_flows)

        for _ in range(num_links):
            start_node = choice(nodes, size = None, p = flows)

            next_elem_probabilities = self.predict_next_element_probabilities([start_node], include_self_links = include_self_links)
            next_nodes, next_probabilities = zip(*next_elem_probabilities.items())

            target_node = choice(next_nodes, size = None, p = next_probabilities)

            link = (start_node, target_node)
            if not link in res:
                res[link] = 0
            res[link] += 1

        return res
