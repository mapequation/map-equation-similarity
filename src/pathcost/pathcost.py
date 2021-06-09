from __future__ import annotations

from infomap     import Infomap
from typing      import List

from .codebook   import CodeBook
from .io.reader  import *

class PathCost():
    """
    PathCost derives codebooks from the modular structure of a network and uses
    them to caluclate the cost in bits to
    """
    def __init__(self) -> None:
        """
        Initialise.
        """

    def from_infomap(self, infomap: Infomap) -> PathCost:
        """
        Construct codebooks from the supplied infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
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
        partition      = PartitionFromInfomap(self.infomap)
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
        if u == v:
            raise Exception("Those are the same nodes, don't do this.")

        return self.cb.get_path_cost_forward(self.addresses[u]) \
             + self.cb.get_walk_cost(self.addresses[u], self.addresses[v])

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
        return 0.5 * ( self.get_path_cost_directed(u, v)
                     + self.get_path_cost_directed(v, u)
                     )
    
    def get_address(self, path: Tuple[str, ...]) -> Tuple[int, ...]:
        """
        """
        # if we don't use memory, we forget all history and simply
        # get the address of the last node in the pathj
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
    
    def predict_next_element_probabilities(self, path: Tuple[str, ...]) -> Dict[str, float]:
        """
        Returns a dictionary with node labels as keys and the probability that
        the respective node is the next node as values.

        Parameters
        ----------
        path: Tuple[str, ...]
            The path to the start node.
        
        Returns
        -------
        Dict[str, float]
            A dictionary with node labels as keys and the probability that
            the respective node is the next node as values.
        """
        if len(path) == 0:
            raise Exception("Predictions with empty path not implemented.")

        source_address = self.get_address(path)

        return { node_label : sum([ self.cb.get_walk_probability(source = source_address, target = target_address)
                                        for target_address in addresses.values()
                                        if  target_address != source_address
                                  ])
                     for node_label, addresses in self.addresses.items()
               }


# class PathCostWithStateNodes(PathCost):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def run_infomap( self
#                    , netfile: str
#                    , directed: bool
#                    , trials: Optional[int]
#                    , seed: Optional[int]
#                    , max_order: Optional[int]
#                    , move_to_lower_order: Optional[bool]
#                    ) -> PathCost:
#         super().run_infomap( netfile  = netfile
#                            , directed = directed
#                            , trials   = trials
#                            , seed     = seed
#                            )

#         # build a dictionary to remember which nodes are valid next steps
#         self._build_constraints( netfile             = netfile
#                                , directed            = directed
#                                , max_order           = max_order
#                                , move_to_lower_order = move_to_lower_order
#                                )
        
#         return self
    
#     def _build_constraints( self
#                           , netfile: str
#                           , directed: bool
#                           , max_order: int
#                           , move_to_lower_order: bool
#                           ) -> None:
#         """
#         Private method for constructing constraints of transitions
#         that respect state histories.

#         Parameters
#         ----------
#         netfile: str
#             The file that contains the network.

#         directed: bool
#             Whether the network is directed.
        
#         max_order: int
#             The maximum order to consider. Relevant for the allowed next states.
#         """
#         network                                       = NetworkFromStateFile(netfile, directed)
#         self.node_IDs_to_node_labels : Dict[int, str] = network.get_nodes()
#         self.state_IDs_to_node_IDs   : Dict[int, int] = { stateID : values["nodeID"] 
#                                                               for (stateID, values) in network.get_state_nodes().items() 
#                                                         }

#         # a mapping from physical nodes to their state nodes where
#         # paths can start
#         self.start_nodes : Dict[int, int] = dict()
#         for (stateID, values) in network.get_state_nodes().items():
#             # nodes where paths can start have the empty history
#             if "{}" in values["label"]:
#                 self.start_nodes[values["nodeID"]] = stateID

#         # extract the memory that corresponds to the state nodes
#         self.state_ID_to_memory : Dict[int, Tuple[str, ...]] = dict()
#         self.memory_to_state    : Dict[Tuple[str, ...], int] = dict()
#         for (stateID, values) in network.get_state_nodes().items():
#             # assuming that state labels are of the form {history}_nodeID
#             # where the history is a sequence of physical node labels, including
#             # the empty label eps, separated by dashes "-".
#             # For example
#             #   1. the label for a state node with empty history in pyhsical node
#             #      with label 42:
#             #        {eps}_42
#             #   2. the label for a state node with 1-step history that passed most
#             #      recently through the physical node with label 47 in physical
#             #      node with label 42:
#             #        {47}_42
#             #   3. the label for a state node with 2-step history that passed most
#             #      recently through the physical nodes with labels 47 and 51 in
#             #      physical node with label 42:
#             #        {47-51}_42
#             memory, current_node_label = values["label"].split("_")
#             # don't include the empty history
#             memory                                               : Tuple[str, ...] = tuple([m for m in memory.strip("{}").split("-") if m != ""])
#             self.state_ID_to_memory[stateID]                                       = memory
#             self.memory_to_state[memory + (current_node_label,)]                   = stateID
        
#         # a mapping from state node IDs to valid next state node IDs
#         self.valid_next_state : Dict[int, List[int]] = { stateID : list() for stateID in self.state_ID_to_memory.keys() }

#         # check which states are possible next steps, these are the states
#         # that respect the history
#         # For example, {42}_47 is a valid next state after {eps}_42, but not
#         # {51}_47.
#         for (current_state_ID, current_state_memory) in self.state_ID_to_memory.items():
#             physical_label = self.node_IDs_to_node_labels[self.state_IDs_to_node_IDs[current_state_ID]]

#             # when we allow moving to a lower order, then all suffixes of the 
#             # current memory, plus the current physical node, are valid next
#             # memories, including the empty memory, that is an initial/terminal node.
#             if move_to_lower_order:
#                 valid_next_memories = [ next_memory
#                                             for next_memory in suffixes(current_state_memory + (physical_label,))
#                                             if len(next_memory) <= max_order
#                                       ]
            
#             # otherwise, we will always stay in the curren order of memory or
#             # move up
#             else:
#                 valid_next_memories = [ next_memory
#                                             for next_memory in [ current_state_memory[1:] + (physical_label,)
#                                                                , current_state_memory     + (physical_label,)
#                                                                ]
#                                             if len(next_memory) <= max_order
#                                       ]

#             for (next_state_stateID, next_state_memory) in self.state_ID_to_memory.items():
#                 # transitions between state nodes inside the same physical node are not allowed
#                 if next_state_stateID != current_state_ID and next_state_memory in valid_next_memories:
#                     self.valid_next_state[current_state_ID].append(next_state_stateID)
    
#     def predict_path(self, start_node: int, steps: int) -> List[int]:
#         """
#         Predicts `steps` many steps for a path starting at `start_node`.
#         We assume that we have an oracle that can tell us how long the
#         path should be so that we can terminate it at the proper length
#         with a transition to an epsilon node.

#         Parameters
#         ----------
#         start_node: int
#             The node where the path starts.
        
#         steps:
#             The number of steps to predict.
#         """
#         res           : List[int] = []
#         current_state : int       = self.start_nodes[start_node]

#         # predict steps many next steps
#         for step in range(1, steps+1):
#             next_state      = None
#             next_state_cost = inf
            
#             # calculate the costs for all valid next state nodes, 
#             # given the current state
#             for candidate in self.valid_next_state[current_state]:
#                 candidate_cost = self.cb.get_walk_cost(self.addresses[current_state], self.addresses[candidate])
                
#                 # select the candidate if it's cheaper to reach
#                 if candidate_cost < next_state_cost:
#                     # we use the path-terminating nodes only in the last step!
#                     if step == steps and next_state in self.start_nodes.values():
#                         next_state      = candidate
#                         next_state_cost = candidate_cost
                    
#                     elif step < steps and next_state not in self.start_nodes.values():
#                         next_state      = candidate
#                         next_state_cost = candidate_cost
            
#             # if we cannot find a next state, we must predict that the path ends here.
#             if next_state is None:
#                 return res
#             else:
#                 res.append(self.state_IDs_to_node_IDs[next_state])
#                 current_state = next_state
        
#         return res
    
#     # ToDo: specify the order
#     def predict_next_element(self, path: Tuple[str, ...]) -> str:
#         """
#         Predict the next element given a `path`.

#         Parameters
#         ----------
#         path: List[int]
#             The observed path.
#         """
#         return self.rank_next_elements(path)[0]
        
    
#     # ToDo: specify the order
#     def rank_next_elements(self, path: Tuple[str, ...]) -> List[str]:
#         """
#         Rank the next elements given a `path`.

#         Parameters
#         ----------
#         path: List[int]
#             The observed path.
#         """
#         history        : Tuple[str, ...] = tuple([self.node_IDs_to_node_labels[node] for node in path])
#         current_state  : int             = self.memory_to_state[history]
#         source_address : Tuple[int, ...] = self.addresses[current_state]
        
#         ranking = []

#         for next_state_candidate in self.valid_next_state[current_state]:
#             target_address = self.addresses[next_state_candidate]
#             target_cost    = self.cb.get_walk_cost(source = source_address, target = target_address)
#             ranking.append((next_state_candidate, target_cost))

#         ranking : List[Tuple[int, float]] = sorted(ranking, key = lambda pair: pair[1])

#         # convert the next state nodes into next physical nodes
#         res : List[str] = []
#         for next_state_candidate, _ in ranking:
#             next_node_candidate = self.state_IDs_to_node_IDs[next_state_candidate]
#             if next_node_candidate not in res:
#                 res.append(next_node_candidate)

#         return res