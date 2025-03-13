from __future__ import annotations
# -----------------------------------------------------------------------------
from infomap                              import Infomap
from itertools                            import islice
from networkx                             import Graph
from matplotlib.bezier                    import BezierSegment
from matplotlib.colors                    import LinearSegmentedColormap
from matplotlib.patches                   import FancyArrowPatch, PathPatch
from matplotlib.path                      import Path
from multiprocessing                      import Pool, cpu_count
from networkx.linalg.laplacianmatrix      import _transition_matrix
from numpy                                import log2, inf as infinity
from numpy.random                         import choice
from typing                               import Any, Dict, Optional as Maybe, Set, Tuple
# -----------------------------------------------------------------------------
from .codebook                            import CodeBook
from .io.reader                           import *
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import seaborn           as sb
# -----------------------------------------------------------------------------


def parallel_jsd_task(args) -> float:
    """
    Task definition for computing JSD per node in parallel.
    """
    ( phi_A
    , phi_B
    , t_A
    , t_B
    , flows_A
    , flows_B
    , p_m_a
    , p_m_b
    , intersection_modules
    , intersection_coding_fraction
    , non_empty_modules_A
    ) = args

    res = 0.0
    for m_a in non_empty_modules_A:
        t_um_a = t_A[m_a]

        for m_b in intersection_coding_fraction[m_a]:
            t_um_b = t_B[m_b]
            v_list = np.array(list(intersection_modules[m_a][m_b]))

            if v_list.shape[0] > 0:
                p_v_A = np.array([flows_A[v] for v in v_list])
                p_v_B = np.array([flows_B[v] for v in v_list])

                cross     = 0.5 * t_um_a * p_v_A / p_m_a[m_a] + 0.5 * t_um_b * p_v_B / p_m_b[m_b]
                log_cross = np.ma.log2(cross).filled(0)

                res -= np.sum(phi_A * t_um_a * p_v_A / p_m_a[m_a] * log_cross)
                res -= np.sum(phi_B * t_um_b * p_v_B / p_m_b[m_b] * log_cross)

    return res


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
                     , filename           : str
                     , generate_codewords : bool = False
                     ) -> MapSim:
        """
        Construct codebooks from an Infomap treefile.

        Parameters
        ----------
        filename : str
            Path to the treefile.

        generate_codewords : bool = False
            Whether to generate codewords.

        Returns
        -------
        MapSim
            The creates MapSim instance.
        """
        partition      : PartitionFromTreeFile                                    = PartitionFromTreeFile(treefile = filename)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : Dict[str, Tuple[int, ...]]                               = partition.get_paths()
        self._build_codebooks(generate_codewords = generate_codewords)
        return self


    def from_infomap( self
                    , infomap            : Infomap
                    , generate_codewords : bool = False
                    ) -> MapSim:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.

        generate_codewords : bool = False
            Whether to generate codewords.

        Returns
        -------
        MapSim
            The creates MapSim instance.
        """
        partition      : PartitionFromInfomap                                     = PartitionFromInfomap(infomap)
        self.modules   : Dict[Tuple[int, ...], Dict[str, Union[Set[int], float]]] = partition.get_modules()
        self.addresses : Dict[str, Tuple[int, ...]]                               = partition.get_paths()
        self._build_codebooks(generate_codewords = generate_codewords)

        # for making predictions later
        self.node_IDs_to_labels : Dict[int, str] = infomap.names

        # pre-compute values for computing flow divergence
        self._precompute_flow_divergence()

        # prepare for more efficient similarity calculations
        # self._prepare_sampling()

        return self


    def from_partition( self
                      , partition          : Partition
                      , generate_codewords : bool = False
                      ) -> MapSim:
        """
        Construct codebooks from the supplied partition.

        Parameters
        ----------
        partition : Partition
            The partition.

        generate_codewords : bool = False
            Whether to generate codewords.

        Returns
        -------
        MapSim
            The creates MapSim instance.
        """
        self.modules   = partition.get_modules()
        self.addresses = partition.get_paths()
        self._build_codebooks(generate_codewords = generate_codewords)
        return self

    def run_infomap( self
                   , netfile                   : str
                   , directed                  : bool
                   , teleportation_probability : float
                   , trials                    : int  = 1
                   , seed                      : int  = 42
                   , one_level                 : bool = False
                   , rawdir                    : bool = False
                   , generate_codewords        : bool = False
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

        generate_codewords : bool = False
            Whether to generate codewords.

        Returns
        -------
        MapSim
            The creates MapSim instance.
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
        self._build_codebooks(generate_codewords = generate_codewords)

        # for making predictions later
        self.node_IDs_to_labels : Dict[int, str] = self.infomap.names

        # pre-compute values for computing flow divergence
        self._precompute_flow_divergence()

        # prepare for more efficient similarity calculations
        # self._prepare_sampling()

        return self


    def _build_codebooks( self
                        , generate_codewords : bool
                        ) -> None:
        """
        Private method for populating the codebooks after loading data.

        Parameters
        ----------
        generate_codewords : bool
            Whether to assign concrete codewords in the codebooks.
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

        # Assigning codewords is technically just a gimmick, 
        # we don't _need_ them but can create them if we _want_ them.
        if generate_codewords:
            self.cb.mk_codewords()


    def _precompute_flow_divergence(self) -> None:
        """
        Precompute terms for efficiently computing flow divergence.
        """
        self.module_transition_rates = dict() # ToDo: they are not module transition rates but mapsim values, that's why m1 == m2 => 1.0
        self.module_coding_fraction  = dict()
        self.module_internal_entropy = dict()
        self.phi                     = dict()

        # cache which nodes are in which modules
        self.non_empty_modules = dict()
        for addr_u in self.addresses.values():
            addr_m = addr_u[:-1]
            if addr_m not in self.non_empty_modules:
                self.non_empty_modules[addr_m] = []
            self.non_empty_modules[addr_m].append(addr_u)

        for m1 in self.non_empty_modules:
            self.module_transition_rates[m1] = dict()
            for m2 in self.non_empty_modules:
                if m1 != m2:
                    # we append 0 to the starting address because we need to consider exiting that module
                    self.module_transition_rates[m1][m2] = self.cb.get_walk_rate((m1) + (0,), m2)
                else:
                    # "similarity" of m1 and m2, here we put the value 1 because it's the neutral element for the logarithm
                    self.module_transition_rates[m1][m2] = 1.0

        for addr_m in self.non_empty_modules:
            m   = self.modules[addr_m]
            p_m = m["flow"] + m["exit"]
            self.module_coding_fraction[addr_m]  = (1.0 - m["exit"] / p_m) if p_m > 0.0 else 1.0
            self.module_internal_entropy[addr_m] = 0.0
            for addr_u in self.non_empty_modules[addr_m]:
                p_u            : float = self.modules[addr_u]["flow"]
                p_u_normalised         = (p_u / p_m) if p_u > 0.0 else 0.0

                self.module_internal_entropy[addr_m] += (p_u_normalised * log2(p_u_normalised)) if p_u_normalised > 0.0 else 0.0

        for addr_u in self.addresses.values():
            numerator = 0
            p_u       = self.modules[addr_u]["flow"]
            for addr_m in self.non_empty_modules:
                numerator += self.module_coding_fraction[addr_m] * self.module_transition_rates[addr_u[:-1]][addr_m]
            self.phi[addr_u] = p_u / numerator if numerator > 0 else 0


    def _precompute_intersection( self
                                , other : MapSim
                                ) -> Tuple[Dict, Dict, Dict, Dict, Dict]: # ToDo: more specific return type
        """
        Compute intersections between partitions for efficient flow divergence computation.

        Parameters
        ----------
        other : MapSim
            The other partition.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict, Dict]
            The precomputed terms.
        """
        # for notational consistency with the paper
        A = self
        B = other

        # remember the intersected modules in the other partition so we don't need to loop over irrelevant modules later
        intersection_coding_fraction    = { m_a : dict() for m_a in self.non_empty_modules }
        intersection_internal_entropies = { m_a : dict() for m_a in self.non_empty_modules }

        # the intersected modules
        intersection_modules = { m_a : dict() for m_a in A.non_empty_modules }

        # for convenience, cache module rates for both partitions
        p_m_a : Dict[Tuple[int,...], float] = dict()
        p_m_b : Dict[Tuple[int,...], float] = dict()

        # pre-compute intersections in a single linear pass over the nodes
        # each node has a membership in this partition and in the other partition,
        # which allows us computing intersections by considering each node only once.
        for u, addr_u_a in self.addresses.items():
            p_u_A    = A.modules[addr_u_a]["flow"]
            addr_u_b = B.addresses[u]
            p_u_B    = B.modules[addr_u_b]["flow"] # for compatibility with different topologies

            m_a = addr_u_a[:-1]
            m_b = addr_u_b[:-1]

            if not m_a in p_m_a:
                p_m_a[m_a] = A.modules[m_a]["flow"] + A.modules[m_a]["exit"]
            if not m_b in p_m_b:
                p_m_b[m_b] = B.modules[m_b]["flow"] + B.modules[m_b]["exit"]

            if m_b not in intersection_internal_entropies[m_a]:
                intersection_internal_entropies[m_a][m_b] = 0
            intersection_internal_entropies[m_a][m_b] += (p_u_A / p_m_a[m_a]) * log2(p_u_B / p_m_b[m_b])

            if m_b not in intersection_coding_fraction[m_a]:
                intersection_coding_fraction[m_a][m_b] = 0
            intersection_coding_fraction[m_a][m_b] += p_u_A / p_m_a[m_a]

            # also remember which nodes sit where
            if m_b not in intersection_modules[m_a]:
                intersection_modules[m_a][m_b] = set()
            intersection_modules[m_a][m_b].add(u)

        return ( p_m_a                            # module flows in partition A
               , p_m_b                            # module flows in partition B
               , intersection_modules             # intersections of modules between A and B
               , intersection_coding_fraction     # coding fractions in the intersected modules
               , intersection_internal_entropies  # internal entropies in the intersected modules
               )


    def _prepare_sampling(self) -> None:
        """
        Precompute terms for generating a network from the partition.
        """
        self.module_transition_rates = dict()
        self.softmax_normalisers     = dict()

        # calculate the transition rates for modules
        modules = [path for path, m in self.modules.items() if len(m["nodes"]) == 0 and len(path) > 0]

        for m1 in modules:
            self.module_transition_rates[m1] = dict()
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
                    self.softmax_normalisers[path] += self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])
                else:
                    self.softmax_normalisers[path] += self.module_transition_rates[path][path_v[:-1]] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])

    def get_probability(self, u, v) -> float:
        """
        Get the probabilits for transitions from u to v.
        """
        path_u = self.addresses[u]
        path_v = self.addresses[v]

        if path_u[:-1] == path_v[:-1]:
            d_uv = self.cb.get_walk_rate(path_u, path_v)
        else:
            d_uv = self.module_transition_rates[path_u[:-1]][path_v[:-1]] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])

        normaliser = self.softmax_normalisers[path_u[:-1]] - self.cb.get_module(path_u[:-1]).get_path_rate_forward(path_u[-1:])
        return d_uv / normaliser


    def mapsim(self, u : str, v : str) -> float:
        """
        MapSim from u to v.
        """
        return self.cb.get_walk_rate(self.addresses[u], self.addresses[v])


    def mapsim2(self, u : str, v : str) -> float:
        """
        MapSim from u to v. More efficient in deep hierarchies.
        """
        addr_u = self.addresses[u]
        addr_v = self.addresses[v]
        return self.module_transition_rates[addr_u[:-1]][addr_v[:-1]] * self.modules[addr_v]["flow"] / self.modules[addr_v]["flow"]


    def plot_hierarchy( self
                      , G                 : Graph
                      , figsize           : Tuple[float, float] = (5,5)
                      , num_spline_points : int                 = 10
                      , hide_links        : bool                = False
                      , link_alpha        : float               = 0.5
                      ) -> None:
        """
        Plot the partition's hierarchical organisation in a circular plot.

        Parameters
        ----------
        G : Graph
            The graph that contains the links.

        figsize : Tuple[float, float]
            The size of the figure.

        num_spline_points : int
            The number of points to draw smooth splines between nodes.
        
        hide_links : bool = False
            Whether to hide the links.

        link_alpha : float = 0.5
            Alpha value for drawing links.
        """
        # node addresses to flow
        node_to_flow = { address : self.cb.get_flow(address) for address in self.addresses.values() }

        # a reverse mapping from addresses to nodes
        address_to_node = { v:k for k,v in self.addresses.items()}

        # remember what colours the nodes get
        address_to_colour = dict()
        address_to_colour[()] = "grey"

        # the nodes that represent modules, as opposed to actual nodes
        module_nodes = set()

        for address in self.addresses.values():
            for init in inits(address):
                module_nodes.add(init)

        # the flows for the module nodes
        module_node_to_flow = dict()
        for module_node in module_nodes:
            sub_codebook = self.cb.get_module(module_node)
            module_node_to_flow[module_node] = sub_codebook.flow

        # the actual plotting...
        fig, ax = plt.subplots(1, 1, figsize = figsize)

        palette = sb.color_palette("colorblind")

        # radial positions for all nodes
        radial_pos = dict()
        radial_pos[()] = (0,0)

        # calculate node positions on the disc
        def child_poincare(x,y,r,theta):
            x_ = x + r * np.cos(theta)
            y_ = y + r * np.sin(theta)

            return (x_,y_)

        # the nodes' modules
        modules = dict()

        theta = 0
        node_colours = []
        node_flows   = []
        for (address, flow) in node_to_flow.items():
            node = address_to_node[address]

            # super-crude way to decide what module the node belongs to
            modules[node] = address[0]
            module = address[0]

            theta += flow * np.pi
            p = child_poincare(0, 0, r = 2, theta = theta)
            radial_pos[address] = p
            theta += flow * np.pi

            node_flows.append(flow)
            node_colours.append(palette[(module-1) % len(palette)])
            address_to_colour[address] = palette[(module-1) % len(palette)]

        ax.pie( node_flows
            , radius = 2.1
            , colors = node_colours
            , wedgeprops = dict( width = 0.1, edgecolor = "w" )
            )

        plt.scatter([0], [0], marker = "s", c = ["grey"])

        angle_offsets = {():0}
        for address in sorted(module_nodes, key = lambda addr: (len(addr), addr)):
            # get angle offset for *this* node
            theta = angle_offsets[address[:-1]]

            # and remember the offset for potential children
            angle_offsets[address] = theta

            theta += module_node_to_flow[address] * np.pi
            r = sum([1/2**i for i in range(len(address))])
            p = child_poincare(0, 0, r = r, theta = theta)
            radial_pos[address] = p
            theta += module_node_to_flow[address] * np.pi

            # and update the angle offset for siblings
            angle_offsets[address[:-1]] = theta

            parent = radial_pos[address[:-1]]
            address_to_colour[address] = palette[(address[0] - 1) % len(palette)]

            plt.plot([parent[0],p[0]], [parent[1],p[1]], c = "grey", alpha = 0.5)
            plt.scatter([p[0]], [p[1]], marker = "s", c = [palette[(address[0] - 1) % len(palette)]])

        if not hide_links:
            for (u, v) in G.edges:
                if u != v:
                    source = self.addresses[u]
                    target = self.addresses[v]
                    path = address_path(source = list(source), target = list(target))
                    points = np.array([radial_pos[tuple(address)] for address in path])
                    bps = bspline(points, n = num_spline_points, degree = 3)

                    # interpolate colours between source and target node
                    cm = LinearSegmentedColormap.from_list("Custom", [address_to_colour[tuple(addr)] for addr in path], N = num_spline_points)

                    for (ix, (p,q)) in enumerate(zip(bps, bps[1:])):
                        frac = ix / len(bps)
                        plt.plot( [p[0], q[0]], [p[1], q[1]], color = cm(frac), alpha = link_alpha)

                    # ax.add_patch(PathPatch(Path(vertices = points, codes = [Path.MOVETO] + (len(points) - 1) * [Path.CURVE3]), fc = "None"))

        ax.axis("off")
        plt.autoscale()
        plt.show()


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


    def D_naive( self
               , other       : MapSim
               , G           : Maybe[Graph] = None
               , samples     : Maybe[int]   = None
               , sample_mode : str          = "uniform"
               , verbose     : bool         = False
               ) -> float:
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
            if samples is not None:
                gens  = dict()

                nodes = []
                flows = []
                for u, addr_u in Ma.addresses.items():
                    nodes.append(u)
                    flows.append(self.cb.get_flow(addr_u))

                r_u_v_a = defaultdict(lambda: dict())
                r_u_v_b = defaultdict(lambda: dict())

                for _ in range(samples):
                    if sample_mode == "uniform":
                        u = np.random.choice(nodes)
                        v = np.random.choice(nodes)
                        while v == u:
                            v = np.random.choice(nodes)
                    elif sample_mode == "biased":
                        u = np.random.choice(nodes, p = flows)
                        v = np.random.choice(nodes, p = flows)
                        while v == u:
                            v = np.random.choice(nodes, p = flows)
                    elif sample_mode == "recommend":
                        u = np.random.choice(nodes, p = flows)
                        if u not in gens:
                            gens[u] = Ma.make_recommendations(u)
                        v = next(gens[u])
                    else:
                        raise Exception(f"Don't know this sampling mode: {sample_mode}")

                    r_u_v_a[u][v] = Ma.cb.get_walk_rate(source = Ma.addresses[u], target = Ma.addresses[v])
                    r_u_v_b[u][v] = Mb.cb.get_walk_rate(source = Mb.addresses[u], target = Mb.addresses[v])

                for u, vs in r_u_v_a.items():
                    p_u = Ma.cb.get_flow(Ma.addresses[u])
                    s_a = sum(vs.values())
                    for v in vs.keys():
                        res += p_u * (r_u_v_a[u][v] / s_a) * log2(r_u_v_a[u][v] / r_u_v_b[u][v])

            else:
                for u, addr_u in Ma.addresses.items():
                    p_u = Ma.cb.get_flow(addr_u)

                    # We need to include self-links here to stay true to the map equation.
                    # Otherwise it means that we remember the current node, but we're only
                    # allowed to remember the current module.
                    r_u_a = Ma.predict_interaction_rates(u, include_self_links = True)
                    r_u_b = Mb.predict_interaction_rates(u, include_self_links = True)
                    s_a   = sum(r_u_a.values())

                    for v in r_u_a.keys():
                        c = (p_u * (r_u_a[v] / s_a) * log2(r_u_a[v] / r_u_b[v])) if r_u_a[v] > 0 and r_u_b[v] > 0 else 0
                        res += c

                        if verbose:
                            print(f"{u}->{v} {p_u:.2f} * {r_u_a[v] / s_a:.2f} * log2({r_u_a[v]:.2f} / {r_u_b[v]:.2f}) = {c:.2f}")

        return res


    def D_per_node_naive( self
                        , other   : MapSim
                        , u       : Any
                        , verbose : bool   = False
                        ) -> float:
        """
        Compute the flow divergence contribution for a single node in a naive way.
        We simply loop over all target nodes to sum up the contribution from the
        source node to all target nodes.

        Parameters
        ----------
        other: MapSim
            The other partition.

        u: Any
            The source node.

        verbose: bool = False
            Whether to be chatty.

        Returns
        -------
        float
            The flow divergence contribution for node u.
        """
        A   = self
        B   = other
        res = 0

        addr_u = A.addresses[u]
        p_u    = A.cb.get_flow(addr_u)

        # We need to include self-links here to stay true to the map equation.
        # Otherwise it means that we remember the current node, but we're only
        # allowed to remember the current module.
        r_u_a = A.predict_interaction_rates(u, include_self_links = True)
        r_u_b = B.predict_interaction_rates(u, include_self_links = True)
        s_a   = sum(r_u_a.values())

        for v in r_u_a.keys():
            c = (p_u * (r_u_a[v] / s_a) * log2(r_u_a[v] / r_u_b[v])) if r_u_a[v] > 0 and r_u_b[v] > 0 else 0
            res += c

            if verbose:
                print(f"{u}->{v} {p_u:.4f} * {r_u_a[v] / s_a:.4f} * log2({r_u_a[v]:.4f} / {r_u_b[v]:.4f}) = {c:.4f}")

        return np.round(res, decimals = 14)


    def D(self, other : MapSim) -> float:
        """
        Compute the flow divergence between this partition and the other one efficiently.

        Parameters
        ----------
        other: MapSim
            The other partition.

        verbose: bool
            Ignored for now.

        Returns
        -------
        float
            The flow divergence between this and the other partition.
        """
        # for notational consistency with the paper
        A = self
        B = other

        ( _p_m_a
        , _p_m_b
        , _intersection_modules
        , intersection_coding_fraction
        , intersection_internal_entropies
        ) = self._precompute_intersection(other = B)

        res = 0

        for u, addr_u in A.addresses.items():
            for m_a in A.non_empty_modules:
                t_um_a = A.module_transition_rates[addr_u[:-1]][m_a]
                res   += (A.phi[addr_u] * t_um_a * (A.module_coding_fraction[m_a] * log2(t_um_a) + A.module_internal_entropy[m_a])) if t_um_a > 0 else 0.0

                for m_b in intersection_coding_fraction[m_a]:
                    t_um_b = B.module_transition_rates[B.addresses[u][:-1]][m_b]
                    res   -= A.phi[addr_u] * t_um_a * (intersection_coding_fraction[m_a][m_b] * log2(t_um_b) + intersection_internal_entropies[m_a][m_b]) if (t_um_a > 0 and t_um_b > 0) else 0.0

        return res


    def D_per_node( self
                  , other   : MapSim
                  , u       : Any
                  ) -> float:
        """
        Compute the flow divergence contribution for a single node efficiently.

        Parameters
        ----------
        other: MapSim
            The other partition.

        u: Any
            The source node.

        Returns
        -------
        float
            The flow divergence contribution for node u.
        """
        # for notational consistency with the paper
        A = self
        B = other

        ( _p_m_a
        , _p_m_b
        , _intersection_modules
        , intersection_coding_fraction
        , intersection_internal_entropies
        ) = self._precompute_intersection(other = B)

        res    = 0
        addr_u = A.addresses[u]

        for m_a in A.non_empty_modules:
            t_um_a = A.module_transition_rates[addr_u[:-1]][m_a]
            res   += A.phi[addr_u] * t_um_a * (A.module_coding_fraction[m_a] * log2(t_um_a) + A.module_internal_entropy[m_a])

            for m_b in intersection_coding_fraction[m_a]:
                t_um_b = B.module_transition_rates[B.addresses[u][:-1]][m_b]
                res   -= A.phi[addr_u] * t_um_a * (intersection_coding_fraction[m_a][m_b] * log2(t_um_b) + intersection_internal_entropies[m_a][m_b])

        return res


    def JSD_naive( self
                 , other   : MapSim
                 , G       : Maybe[Graph] = None
                 , verbose : bool         = False
                 ) -> float:
        A = self
        B = other

        res : float = 0.0
        for u, addr_u_A in A.addresses.items():
            addr_u_B : Tuple[int, ...] = B.addresses[u]
            p_u_A    : float           = A.modules[addr_u_A]["flow"]
            p_u_B    : float           = B.modules[addr_u_B]["flow"]

            mapsims_a = { v:A.mapsim(u, v) for v in A.addresses.keys() }
            mapsims_b = { v:B.mapsim(u, v) for v in B.addresses.keys() }

            sum_a = sum(mapsims_a.values())
            sum_b = sum(mapsims_b.values())

            for v in A.addresses.keys():
                a = (p_u_A * (mapsims_a[v] / sum_a) * log2(mapsims_a[v] / (0.5 * mapsims_a[v] + 0.5 * mapsims_b[v]))) if mapsims_a[v] > 0 else 0.0
                b = (p_u_B * (mapsims_b[v] / sum_b) * log2(mapsims_b[v] / (0.5 * mapsims_a[v] + 0.5 * mapsims_b[v]))) if mapsims_b[v] > 0 else 0.0
                res += 0.5 * (a + b)

        return np.sqrt(res)

        # class Mixture:
        #     def __init__(self, A, B):
        #         self.A = A
        #         self.B = B

        #     def predict_interaction_rates(self, u, include_self_links : bool = True) -> Dict[str, float]:
        #         r_u_a = self.A.predict_interaction_rates(u, include_self_links = True)
        #         r_u_b = self.B.predict_interaction_rates(u, include_self_links = True)
        #         return { v : 1/2 * r_u_a[v] + 1/2 * r_u_b[v] for v in r_u_a.keys() }

        # M = Mixture(A, B)

        # return np.round(1/2 * A.D_naive(M) + 1/2 * B.D_naive(M), decimals = 14)

    def JSD_sample(self, other : MapSim, k : int = 10) -> float:
        A = self
        B = other

        res : float = 0.0
        # sample the k most important target nodes from each source node
        for u, addr_u_A in A.addresses.items():
            addr_u_B  : Tuple[int, ...] = B.addresses[u]
            p_u_A     : float           = A.modules[addr_u_A]["flow"]
            p_u_B     : float           = B.modules[addr_u_B]["flow"]

            mapsims_a = { v:A.mapsim(u, v) for v in islice(A.make_recommendations(u), k) }
            mapsims_b = { v:B.mapsim(u ,v) for v in mapsims_a.keys()                     }

            sum_a = sum(mapsims_a.values())
            sum_b = sum(mapsims_b.values())

            for v in mapsims_a.keys():
                a        = (p_u_A * (mapsims_a[v] / sum_a) * log2(mapsims_a[v] / (0.5 * mapsims_a[v] + 0.5 * mapsims_b[v]))) if mapsims_a[v] > 0 else 0.0
                b        = (p_u_B * (mapsims_b[v] / sum_b) * log2(mapsims_b[v] / (0.5 * mapsims_a[v] + 0.5 * mapsims_b[v]))) if mapsims_b[v] > 0 else 0.0
                res += 0.5 * (a + b)

        return np.sqrt(res)


    def JSD(self, other : MapSim) -> float:
        A = self
        B = other

        ( p_m_a
        , p_m_b
        , intersection_modules
        , intersection_coding_fraction
        , intersection_internal_entropies
        ) = self._precompute_intersection(other = B)

        res_A_M : float = 0.0
        res_B_M : float = 0.0

        # We compute the result in three steps
        # d_F(A,B) = sqrt(1/2 * D_F(A||M) + 1/2 * D_F(B||M))

        # Part for A
        for u, addr_u_A in A.addresses.items():
            for m_a in A.non_empty_modules:
                t_um_a = A.module_transition_rates[addr_u_A[:-1]][m_a]
                if t_um_a > 0.0:
                    res_A_M += A.phi[addr_u_A] * t_um_a * (A.module_coding_fraction[m_a] * log2(t_um_a) + A.module_internal_entropy[m_a])

        # Part for B
        for u, addr_u_B in B.addresses.items():
            for m_b in B.non_empty_modules:
                t_um_b = B.module_transition_rates[addr_u_B[:-1]][m_b]
                if t_um_b > 0.0:
                    res_B_M += B.phi[addr_u_B] * t_um_b * (B.module_coding_fraction[m_b] * log2(t_um_b) + B.module_internal_entropy[m_b])

        # Part for A and B
        # unfortunately, this part is not efficient (yet)
        for u, addr_u_A in A.addresses.items():
            addr_u_B = B.addresses[u]
            phi_A    = A.phi[addr_u_A]
            phi_B    = B.phi[addr_u_B]

            for m_a in A.non_empty_modules:
                t_um_a    = A.module_transition_rates[addr_u_A[:-1]][m_a]

                for m_b in intersection_coding_fraction[m_a]:
                    t_um_b    = B.module_transition_rates[addr_u_B[:-1]][m_b]

                    # Get all nodes in the intersection as a NumPy array
                    v_list = np.array(list(intersection_modules[m_a][m_b]))

                    if v_list.shape[0] > 0:
                        # Vectorized access to module "flow" values
                        p_v_A = np.array([A.modules[A.addresses[v]]["flow"] for v in v_list])
                        p_v_B = np.array([B.modules[B.addresses[v]]["flow"] for v in v_list])

                        # Compute cross probabilities as a NumPy array
                        cross     = 0.5 * t_um_a * p_v_A / p_m_a[m_a] + 0.5 * t_um_b * p_v_B / p_m_b[m_b]
                        log_cross = np.ma.log2(cross).filled(0)  # Vectorized log computation

                        # Compute vectorized results
                        res_A_M -= np.sum(phi_A * t_um_a * p_v_A / p_m_a[m_a] * log_cross)
                        res_B_M -= np.sum(phi_B * t_um_b * p_v_B / p_m_b[m_b] * log_cross)

                        # for v in intersection_modules[m_a][m_b]:
                        #     p_v_A = A.modules[A.addresses[v]]["flow"]
                        #     p_v_B = B.modules[B.addresses[v]]["flow"]
                        #     cross = log2 (0.5 * t_um_a * p_v_A / p_m_a[m_a]
                        #                 + 0.5 * t_um_b * p_v_B / p_m_b[m_b]
                        #                 )
                        #     res_A_M -= phi_u_A * t_um_a * p_v_A / p_m_a[m_a] * cross
                        #     res_B_M -= phi_u_B * t_um_b * p_v_B / p_m_b[m_b] * cross

        return np.sqrt(0.5 * res_A_M + 0.5 * res_B_M)


    def JSD_parallel( self
                    , other  : MapSim
                    , n_jobs : Maybe[int] = None
                    ) -> float:
        if n_jobs is None:
            n_jobs = cpu_count()

        A = self
        B = other

        ( p_m_a
        , p_m_b
        , intersection_modules
        , intersection_coding_fraction
        , intersection_internal_entropies
        ) = self._precompute_intersection(other = B)

        res_A_M : float = 0.0
        res_B_M : float = 0.0

        # We compute the result in three steps
        # d_F(A,B) = sqrt(1/2 * D_F(A||M) + 1/2 * D_F(B||M))

        # Part for A
        for u, addr_u_A in A.addresses.items():
            for m_a in A.non_empty_modules:
                t_um_a = A.module_transition_rates[addr_u_A[:-1]][m_a]
                if t_um_a > 0.0:
                    res_A_M += A.phi[addr_u_A] * t_um_a * (A.module_coding_fraction[m_a] * log2(t_um_a) + A.module_internal_entropy[m_a])

        # Part for B
        for u, addr_u_B in B.addresses.items():
            for m_b in B.non_empty_modules:
                t_um_b = B.module_transition_rates[addr_u_B[:-1]][m_b]
                if t_um_b > 0.0:
                    res_B_M += B.phi[addr_u_B] * t_um_b * (B.module_coding_fraction[m_b] * log2(t_um_b) + B.module_internal_entropy[m_b])

        flows_A = { v : A.modules[addr_v_A]["flow"] for v, addr_v_A in A.addresses.items() }
        flows_B = { v : B.modules[addr_v_B]["flow"] for v, addr_v_B in B.addresses.items() }

        task_parameters = [ ( A.phi[addr_u_A]
                            , B.phi[B.addresses[u]]
                            , A.module_transition_rates[A.addresses[u][:-1]] # only need the transition rates from u's module
                            , B.module_transition_rates[B.addresses[u][:-1]] # only need the transition rates from u's modules
                            , flows_A
                            , flows_B
                            , p_m_a
                            , p_m_b
                            , intersection_modules
                            , intersection_coding_fraction
                            , A.non_empty_modules
                            )
                              for u, addr_u_A in A.addresses.items()
                          ]

        with Pool(processes = n_jobs) as pool:
            res_par = np.sum(pool.map(parallel_jsd_task, task_parameters))

        return np.sqrt(0.5 * res_A_M + 0.5 * res_B_M + 0.5 * res_par)


class MapSimMixture(MapSim):
    def __init__(self, A, B):
        super().__init__()
        raise NotImplementedError("MapSimMixture.__init__")


class OverlappingMapSim(MapSim):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("OverlappingMapSim.__init__")

    def from_treefile(self, filename: str) -> OverlappingMapSim:
        raise Exception("Cannot construct overlapping mapsim from tree file.")

    def from_infomap(self, infomap: Infomap) -> OverlappingMapSim:
        raise Exception("Cannot construct overlapping mapsim from infomap.")

    def from_soft_assignments(self, S, F, p, the_nodes):
        self.partition = PartitionFromSoftAssignmentMatrices(S = S, F = F, p = p, the_nodes = the_nodes)
        self.from_partition(partition = self.partition)
        return self

    def predict_interaction_rates(self, node: str, include_self_links: bool = True) -> Dict[str, float]:
        source_addresses = self.get_address(node)
        source_flow      = sum([self.cb.get_flow(source_address) for source_address in source_addresses])

        rates = dict()

        # start at all parts of the source node
        for source_address in source_addresses:
            partial_source = self.cb.get_flow(source_address) / source_flow
            for target_node, target_addresses in self.addresses.items():
                if target_node != node or include_self_links:
                    target_flow = sum([self.cb.get_flow(target_address) for target_address in target_addresses])
                    rates[target_node] = 0
                    # go to all parts of the target node
                    for target_address in target_addresses:
                        partial_target = self.cb.get_flow(target_address) / target_flow
                        rates[target_node] += partial_source * partial_target * self.cb.get_walk_rate(source = source_address, target = target_address)

        return rates

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
                addrs_u = M.get_address(u)
                for addr_u in addrs_u:
                    # p_u = M.cb.get_flow(M.addresses[u])
                    for ix_v, v in enumerate(G.nodes):
                        addrs_v = M.get_address(v)
                        for addr_v in addrs_v:
                            p_u = M.cb.get_flow(addr_u)
                            if v in G.neighbors(u):
                                c    = - p_u * T[ix_u,ix_v] * log2(M.cb.get_walk_rate(addr_u, addr_v))
                                res += c

                                if verbose:
                                    print(f"{u}->{v} - {p_u:.2f} * {T[ix_u,ix_v]:.2f} * log2({M.mapsim(u, v):.2f}) = {c:.2f}")

        else:
            for u, addrs_u in M.addresses.items():
                for addr_u in addrs_u:
                    p_u = M.cb.get_flow(addr_u)
                    t_u = M.predict_interaction_rates(node = u, include_self_links = False)
                    s   = sum(t_u.values())

                    for v, t_uv in t_u.items():
                        c    = - p_u * (t_uv / s) * log2(t_uv)
                        res += c

                        if verbose:
                            print(f"{u}->{v} - {p_u:.2f} * {t_uv / s:.2f} * log2({t_uv:.2f}) = {c:.2f}")

        return res

    def D(self, other: MapSim, G: Graph | None = None, verbose: bool = False) -> float:
        """
        Use MapSim for efficient computations.
        """
        raise NotImplemented

    def D_naive(self, other : MapSim, verbose : bool = False) -> float:
        A   = self
        B   = other
        res = 0

        for u, addrs_u in A.addresses.items():
            p_u = sum([A.cb.get_flow(addr_u) for addr_u in addrs_u])

            r_u_a = A.predict_interaction_rates(u, include_self_links = False)
            r_u_b = B.predict_interaction_rates(u, include_self_links = False)
            s_a   = sum(r_u_a.values())

            for v in r_u_a.keys():
                c = (p_u * (r_u_a[v] / s_a) * log2(r_u_a[v] / r_u_b[v])) if r_u_a[v] > 0 and r_u_b[v] > 0 else 0
                res += c

                if verbose:
                    print(f"{u}->{v} {p_u:.2f} * {r_u_a[v] / s_a:.2f} * log2({r_u_a[v]:.2f} / {r_u_b[v]:.2f}) = {c:.2f}")

        return res

    def D_per_node(self, other: MapSim, u=None, G: Graph | None = None, verbose: bool = False) -> float:
        """
        Use MapSim for efficient computations.
        """
        raise NotImplemented

    def D_per_node_naive(self, other : MapSim, u, verbose : bool = False) -> float:
        Ma  = self
        Mb  = other
        res = 0

        addrs_u = Ma.addresses[u]
        p_u     = sum([Ma.cb.get_flow(addr_u) for addr_u in addrs_u])

        r_u_a = Ma.predict_interaction_rates(u, include_self_links = False)
        r_u_b = Mb.predict_interaction_rates(u, include_self_links = False)
        s_a   = sum(r_u_a.values())

        for v in r_u_a.keys():
            c = (p_u * (r_u_a[v] / s_a) * log2(r_u_a[v] / r_u_b[v])) if r_u_a[v] > 0 and r_u_b[v] > 0 else 0
            res += c

            if verbose:
                print(f"{u}->{v} {p_u:.2f} * {r_u_a[v] / s_a:.2f} * log2({r_u_a[v]:.2f} / {r_u_b[v]:.2f}) = {c:.2f}")

        return np.round(res, decimals = 14)



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
            self.module_transition_rates[m1] = dict()
            for m2 in modules:
                if m1 != m2:
                    # we append 0 to the starting address because we need to consider exiting that module
                    self.module_transition_rates[m1][m2] = self.cb.get_walk_rate((m1) + (0,), m2)

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
                    self.softmax_normalisers[path] += 2**(self.beta * log2(self.module_transition_rates[path][path_v[:-1]] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:])))

    def from_infomap(self, infomap: Infomap, mapping: Dict[int, str] | None = None, netfile: str | None = None) -> MapSim:
        super().from_infomap(infomap)

        return self

    def get_probability(self, u, v) -> float:
        path_u     = self.addresses[u]
        path_v     = self.addresses[v]
        if path_u[:-1] == path_v[:-1]:
            d_uv = log2(self.cb.get_walk_rate(path_u, path_v))
        else:
            d_uv = log2(self.module_transition_rates[path_u[:-1]][path_v[:-1]] * self.cb.get_module(path_v[:-1]).get_path_rate_forward(path_v[-1:]))
        normaliser = self.softmax_normalisers[path_u[:-1]] - 2**(self.beta * log2(self.cb.get_module(path_u[:-1]).get_path_rate_forward(path_u[-1:])))
        return self.population(u) * 2**(self.beta * d_uv) / normaliser
