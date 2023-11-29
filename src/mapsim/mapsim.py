from __future__ import annotations

from infomap                         import Infomap
from networkx                        import Graph
from networkx.linalg.laplacianmatrix import _transition_matrix
from numpy                           import log2, inf as infinity
from numpy.random                    import choice
from typing                          import Dict, Optional as Maybe, Set, Tuple

from .codebook                       import CodeBook
from .io.reader                      import *

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

    def from_infomap(self, infomap: Infomap) -> MapSim:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
        """
        partition      : PartitionFromInfomap                                     = PartitionFromInfomap(infomap)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : Dict[str, Tuple[int, ...]]                               = partition.get_paths()
        self._build_codebooks()

        # for making predictions later
        self.node_IDs_to_labels : Dict[int, str] = infomap.names

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
        self.cb.mk_codewords()
        

    def mapsim(self, u: str, v : str) -> float:
        return self.cb.get_walk_rate(self.addresses[u], self.addresses[v])

    def mapsim(self, u: str, v : str) -> float:
        return self.cb.get_walk_rate(self.addresses[u], self.addresses[v])

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
        return -log2(walk_rate) if walk_rate > 0 else infinity

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

            next_elem_probabilities = self.predict_interaction_probabilities(start_node, include_self_links = include_self_links)
            next_nodes, next_probabilities = zip(*next_elem_probabilities.items())

            target_node = choice(next_nodes, size = None, p = next_probabilities)

            link = (start_node, target_node)
            if not link in res:
                res[link] = 0
            res[link] += 1

        return res
    
    def L(self, G : Maybe[Graph] = None, verbose : bool = False) -> float:
        """
        Calculate the codelength.

        Parameters
        ----------
        G: Maybe[Graph]
            The graph for calculating transition rates.
            If no graph is given, transition rates are estimated based on
            the modular compression of flows.
        
        verbose : bool
            Print info while calculating.
        
        Returns
        -------
        float
            The codelength.
        """
        M   = self
        res = 0

        if G is not None:
            T = _transition_matrix(G.to_directed()).toarray()
            for ix_u, u in enumerate(G.nodes):
                p_u = M.cb.get_flow(M.addresses[u])
                for ix_v, v in enumerate(G.nodes):
                    if v in G.neighbors(u):
                        c    = - p_u * T[ix_u,ix_v] * log2(M.mapsim(u, v))
                        res += c

                        if verbose:
                            print(f"{u}->{v} - {p_u:.2f} * {T[ix_u,ix_v]:.2f} * log2({M.mapsim(u, v):.2f}) = {c:.2f}")
        
        else:
            for u, addr_u in M.addresses.items():
                p_u = M.cb.get_flow(addr_u)
                t_u = M.predict_interaction_rates(node = u, include_self_links = False)
                s   = sum(t_u.values())

                for v, t_uv in t_u.items():
                    c    = - p_u * (t_uv / s) * log2(t_uv)
                    res += c

                    if verbose:
                        print(f"{u}->{v} - {p_u:.2f} * {t_uv / s:.2f} * log2({t_uv:.2f}) = {c:.2f}")

        return res

    def L_per_node(self, G : Graph) -> Dict[str, float]:
        M   = self
        res = dict()

        T = _transition_matrix(G.to_directed()).toarray()
        for ix_u, u in enumerate(G.nodes):
            res[u] = 0
            p_u = M.cb.get_flow(M.addresses[u])
            for ix_v, v in enumerate(G.nodes):
                if v in G.neighbors(u):
                    res[u] += - p_u * T[ix_u,ix_v] * log2(M.mapsim(u, v))
        
        return res

    def D(self, other : MapSim, G : Maybe[Graph] = None, verbose : bool = False) -> float:
        Ma = self
        Mb = other
        
        res = 0

        if G is not None:
            T = _transition_matrix(G.to_directed()).toarray()
            for ix_u, u in enumerate(G.nodes):
                p_u = Ma.cb.get_flow(self.addresses[u])
                for ix_v, v in enumerate(G.nodes):
                    if v in G.neighbors(u):
                        c    = p_u * T[ix_u,ix_v] * log2(Ma.mapsim(u, v) / Mb.mapsim(u, v))
                        res += c

                        if verbose:
                            print(f"{u}->{v} {p_u:.2f} * {T[ix_u,ix_v]:.2f} * log2({Ma.mapsim(u, v):.2f} / {Mb.mapsim(u, v):.2f}) = {c:.2f}")
        else:
            for u, addr_u in Ma.addresses.items():
                p_u = Ma.cb.get_flow(addr_u)

                t_u_a = Ma.predict_interaction_rates(u, include_self_links = False)
                t_u_b = Mb.predict_interaction_rates(u, include_self_links = False)

                s_a = sum(t_u_a.values())

                for v in t_u_a.keys():
                    c    = p_u * (t_u_a[v] / s_a) * log2(t_u_a[v] / t_u_b[v])
                    res += c

                    if verbose:
                        print(f"{u}->{v} {p_u:.2f} * {t_u_a[v] / s_a:.2f} * log2({t_u_a[v]:.2f} / {t_u_b[v]:.2f}) = {c:.2f}")

        return res


class MapSimSampler(MapSim):
    def __init__(self, G) -> None:
        super().__init__()
        
        if G.is_directed():
            self.population = lambda u: G.out_degree(u)
        else:
            self.population = lambda u: G.degree(u)
    
    def prepare_sampling(self, beta : float) -> None:
        self.beta = beta
        self.module_transition_rates = dict()

        # calculate the transition rates for modules
        modules = [path for path, m in self.modules.items() if len(m["nodes"]) == 0 and len(path) > 0]

        for m1 in modules:
            for m2 in modules:
                if m1 != m2:
                    # we append 0 to the starting address because we need to consider exiting that module
                    self.module_transition_rates[(m1, m2)] = self.cb.get_walk_rate((m1) + (0,), m2)
        
        # calculate the softmax normalisers
        self.softmax_normalisers = dict()
        the_nodes                = set(self.cb.get_nodes())
        modules                  = [path for path, m in self.modules.items() if len(m["nodes"]) == 0 and len(path) > 0]
        for path in modules:
            self.softmax_normalisers[path] = 0

            module_nodes = set(self.cb.get_module(path).get_nodes())
            for v in the_nodes:
                path_v = self.addresses[v]
                if v in module_nodes:
                    self.softmax_normalisers[path] += 2**(self.beta * log2(self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])))
                else:
                    self.softmax_normalisers[path] += 2**(self.beta * log2(self.module_transition_rates[(path, path_v[:-1])] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])))

    def from_infomap(self, infomap: Infomap, mapping: Dict[int, str] | None = None, netfile: str | None = None) -> MapSim:
        super().from_infomap(infomap, mapping, netfile)

        return self

    def get_probability(self, u, v) -> float:
        path_u     = self.addresses[u]
        path_v     = self.addresses[v]
        if path_u[:-1] == path_v[:-1]:
            d_uv = log2(self.cb.get_walk_rate(path_u, path_v))
        else:
            d_uv = log2(self.module_transition_rates[(path_u[:-1], path_v[:-1])] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:]))
        normaliser = self.softmax_normalisers[path_u[:-1]] - 2**(self.beta * log2(self.cb.get_module(path_u[:-1]).get_path_rate_forward(path_u[-1:])))
        return self.population(u) * 2**(self.beta * d_uv) / normaliser
