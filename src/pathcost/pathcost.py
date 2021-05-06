from __future__ import annotations

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

    def from_infomap(self, infomap: Infomap) -> PathCost:
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
        self._build_codebooks()
        return self

    def run_infomap(self, netfile: str, directed: bool, trials: Optional[int] = 1, seed: Optional[int] = 42) -> PathCost:
        """
        Run infomap on the supplied network file and use the partition it finds.

        Parameters
        ----------
        netfile: str
            The file that contains the network.

        directed: bool = False
            Whether the network is directed or not.

        trials: Optional[int]
            Number of trials that infomap should run.
        
        seed: Optional[int]
            The seed for infomap.
        """

        # run infomap
        infomap_args = [f"--silent --num-trials {trials} --seed {seed}"]
        if directed:
            infomap_args.append("--directed")

        self.infomap = Infomap(" ".join(infomap_args))
        self.infomap.read_file(netfile)
        self.infomap.run()

        # extract the partition from infomap
        partition = PartitionFromInfomap(self.infomap)

        self.modules = partition.get_modules()
        self.paths   = partition.get_paths()
        self._build_codebooks()

        # build a dictionary to remember which nodes are valid next steps
        self._build_constraints(netfile = netfile, directed = directed)

        return self

    def _build_codebooks(self) -> None:
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

    def _build_constraints(self, netfile: str, directed: bool) -> None:
        """
        Private method for constructing constraints of transitions
        that respect state histories.
        """
        network                    = NetworkFromStateFile(netfile, directed)
        node_IDs_to_node_labels    = network.get_nodes()
        self.state_IDs_to_node_IDs = { stateID:values["nodeID"] 
                                           for stateID,values in network.get_state_nodes().items() 
                                     }

        # a mapping from physical nodes to their state nodes where
        # paths can start
        self.start_nodes = dict()
        for stateID, values in network.get_state_nodes().items():
            # nodes where paths can start have the empty history
            if "{}" in values["label"]:
                self.start_nodes[values["nodeID"]] = stateID

        # extract the memory that corresponds to the state nodes
        state_memory = dict()
        for stateID, values in network.get_state_nodes().items():
            # assuming that state labels are of the form {history}_nodeID
            # where the history is a sequence of physical node labels, including
            # the empty label eps, separated by dashes "-".
            # For example
            #   1. the label for a state node with empty history in pyhsical node
            #      with label 42:
            #        {eps}_42
            #   2. the label for a state node with 1-step history that passed most
            #      recently through the physical node with label 47 in physical
            #      node with label 42:
            #        {47}_42
            #   3. the label for a state node with 2-step history that passed most
            #      recently through the physical nodes with labels 47 and 51 in
            #      physical node with label 42:
            #        {47-51}_42
            history, current_node_label = values["label"].split("_")
            # don't include the empty history
            history                     = [h for h in history.strip("{}").split("-") if h != ""]
            state_memory[stateID]       = history
        
        # a mapping from state node IDs to valid next state node IDs
        self.valid_next_state = { stateID:list() for stateID in state_memory.keys() }

        # check which states are possible next steps, these are the states
        # that respect the history
        # For example, {42}_47 is a valid next state after {eps}_42, but not
        # {51}_47.
        for current_state_ID, current_state_memory in state_memory.items():
            physical_label       = node_IDs_to_node_labels[self.state_IDs_to_node_IDs[current_state_ID]]
            valid_next_histories = list(suffixes(current_state_memory + [physical_label]))
            for next_state_stateID, next_state_memory in state_memory.items():
                if next_state_stateID != current_state_ID and next_state_memory in valid_next_histories:
                    self.valid_next_state[current_state_ID].append(next_state_stateID)

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
