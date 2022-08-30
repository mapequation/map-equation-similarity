from __future__ import annotations

from infomap      import Infomap
from numpy        import log2
from numpy.random import choice
from typing       import Optional as Maybe, Tuple

from .codebook    import CodeBook
from .io.reader   import *

class MapSim():
    """
    MapSim derives codebooks from the modular structure of a network and uses
    them to caluclate the cost in bits to
    """
    def __init__(self) -> None:
        """
        Initialise.
        """

    def from_treefile( self
                     , filename : str
                     ) -> MapSim:
        partition      : PartitionFromTreeFile                                    = PartitionFromTreeFile(treefile = filename)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : Dict[str, Tuple[int, ...]]                               = partition.get_paths()
        self._build_codebooks()
        return self

    def from_infomap(self, infomap: Infomap, mapping : Maybe[Dict[int, str]] = None, netfile : Maybe[str] = None) -> MapSim:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
        
        mapping: Maybe[Dict[int, str]]
            A mapping from infomap-internal node IDs to node labels.

        netfile: Maybe[str] = none
            The file that contains the network (needed for of memory networks).
        """
        partition      : PartitionFromInfomap                                     = PartitionFromInfomap(infomap, mapping = mapping)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : Dict[str, Tuple[int, ...]]                               = partition.get_paths()
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
                   ) -> MapSim:
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

        # extract the partition from infomap
        partition      = PartitionFromInfomap(self.infomap, mapping = None, netfile = netfile)
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
            self.cb.insert_path( node  = list(self.modules[m]["nodes"])[0] if len(self.modules[m]["nodes"]) == 1 else None
                               , path  = m
                               , flow  = self.modules[m]["flow"]
                               , enter = self.modules[m]["enter"]
                               , exit  = self.modules[m]["exit"]
                               )
        self.cb.calculate_normalisers()
        self.cb.calculate_costs()

    def get_path_cost_directed(self, u: str, v: str) -> float:
        """
        Calculate the path cost for the directed edge (u,v).

        Parameters
        ----------
        u: int
            The source node.

        v: int
            The target node.
        """
        walk_rate = self.cb.get_walk_rate(self.addresses[u], self.addresses[v])
        return -log2(walk_rate) if walk_rate > 0 else numpy.inf

    def get_path_cost_undirected(self, u: str, v: str) -> float:
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

    def get_address(self, node : str) -> Tuple[int, ...]:
        """
        """
        return self.addresses[node]


    def predict_interaction_rates( self
                                 , node: str
                                 , include_self_links : bool = True
                                 ) -> Dict[str, float]:
        """
        Returns a dictionary with node labels as keys and the interaction rates
        with other nodes as values.
        These values are not normalised, that is, they do not necessarily sum to 1.

        Parameters
        ----------
        node: str
            The node for which we want to predict interaction rates.

        include_self_links: bool = True
            Whether to include self-links. To be true to the map equation,
            self-links should be included, otherwise this means that we would
            remember from which node we start -- the map equation does not
            remember this.

        Returns
        -------
        Dict[str, float]
            A dictionary with node labels as keys and the interaction rates
            with other nodes as values.
        """
        source_address = self.get_address(node)

        rates = { node_label : self.cb.get_walk_rate(source = source_address, target = target_address)
                      for node_label, target_address in self.addresses.items()
                      if include_self_links or target_address != source_address
                }

        return rates


    def predict_interaction_probabilities( self
                                         , node               : str
                                         , include_self_links : bool = True
                                         ):
        """
        Returns a dictionary with node labels as keys and the interaction
        probabilities with other nodes as values.

        Parameters
        ----------
        node: str
            The node for which we want to predict interaction probabilities.

        include_self_links: bool = True
            Whether to include self-links. To be true to the map equation,
            self-links should be included, otherwise this means that we would
            remember from which node we start -- the map equation does not
            remember this.

        Returns
        -------
        Dict[str, float]
            A dictionary with node labels as keys and the interaction
            probabilities with other nodes as values.
        """
        rates = self.predict_interaction_rates( node = node
                                              , include_self_links = include_self_links
                                              )
        s = sum(rates.values())
        return { node : rate / s for (node, rate) in rates.items() }


    def make_recommendations( self
                            , node : str
                            ):
        yield from self.cb.recommend(self.addresses[node])


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