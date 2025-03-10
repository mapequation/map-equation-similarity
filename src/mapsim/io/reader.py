from __future__   import annotations
from abc          import ABCMeta, abstractmethod
from collections  import defaultdict
from infomap      import Infomap
from typing       import Callable, DefaultDict, Dict, List, Set, Optional as Maybe, Tuple, Union

from ..util import *

import numpy as np

class Network(metaclass = ABCMeta):
    @abstractmethod
    def get_nodes(self):
        raise NotImplemented

    @abstractmethod
    def get_edges(self):
        raise NotImplemented

class NetworkFromInfomap(Network):
    """
    A reader for networks and partitions that reads from an Infomap instance.
    """
    def __init__(self, infomap: Infomap):
        """
        Initialise the reader.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance to read from.
        """
        self.infomap = infomap

    def get_nodes(self) -> List[int]:
        """Returns the nodes."""
        # assuming for now that we will use state nodes
        return sorted([node.state_id for node in self.infomap.nodes])

    def get_edges(self) -> List[int]:
        """Returns the edges."""
        raise NotImplemented


class NetworkFromNetFileDisk(Network):
    """
    A reader for net files that contain only nodes and edges.
    We scan the net file for the start of nodes and edges so that we can jump
    there when requested instead of storing all nodes and edges in memory.

    This class is NOT thread-safe!
    """
    def __init__(self, filename: str):
        """
        Scan through the file and remember where the section for nodes starts
        and ends, and where the section for edges start.

        Parameters
        ----------
        filename: str
            The net file to read.
        """
        self.fh = open(filename, "r")

        # seek for the start of nodes
        line = self.fh.readline()
        while not line.startswith("*Vertices"):
            line = self.fh.readline()

        # mark the start of nodes
        self.nodeStart = self.fh.tell()

        # seek for the start of edges
        while not (line.startswith("*Edges") or line.startswith("*Links")):
            # mark the end of nodes
            self.nodeEnd = self.fh.tell()
            line = self.fh.readline()

        # mark the start of edges
        self.edgeStart = self.fh.tell()

    def close(self):
        """
        The file handle needs to be closed manually because we can't guess when
        the user is done with it.
        """
        self.fh.close()

    def get_nodes(self):
        """
        Jump to the start of the nodes section in the file and yield the node IDs.
        """
        self.fh.seek(self.nodeStart)
        while self.fh.tell() < self.nodeEnd:
            line   = self.fh.readline()
            nodeID = ""
            i      = 0
            while line[i] != " ":
                nodeID += line[i]
                i      += 1
            yield int(nodeID)

    def get_edges(self):
        """
        Jump to the start of the edges section in the file and yield the edges.
        """
        self.fh.seek(self.edgeStart)
        line = self.fh.readline()
        while line:
            u,v,w = "", "", ""
            i = 0
            while line[i] != " ":
                u += line[i]
                i += 1
            i += 1
            while line[i] != " ":
                v += line[i]
                i += 1
            while line[i] != "\n":
                w += line[i]
                i += 1
            yield (int(u), int(v), float(w))
            line = self.fh.readline()


class NetworkFromNetFile(Network):
    """
    A reader for net files that contain only nodes and edges.
    """
    def __init__(self, filename: str, directed: bool = False):
        """
        Reads the file and stores nodes and edges in memory.

        Parameters
        ----------
        filename: str
            The net file to read.

        directed: bool = False
            Whether the network is directed.
        """
        self.directed : bool                         = directed
        self.nodes    : Dict[int, str]               = dict()
        self.edges    : List[Tuple[int, int, float]] = []

        with open(filename, "r") as fh:
            line = fh.readline()
            while not line.startswith("*Vertices"):
                line = fh.readline()

            line = fh.readline()
            while not (line.startswith("*Edges") or line.startswith("*Links")):
                node_ID, node_label = line.strip().split()
                self.nodes[int(node_ID)] = node_label.strip("\"")
                line = fh.readline()

            line = fh.readline()
            while line:
                edge = line.split()
                if len(edge) == 2:
                    u, v, w = edge[0], edge[1], 1.0
                elif len(edge) == 3:
                    u, v, w = edge[0], edge[1], edge[2]
                else: 
                    raise Exception(f"Unexpected data in edge: {edge}")
                self.edges.append((int(u),int(v),float(w)))
                line = fh.readline()

    def get_nodes(self) -> Dict[int, str]:
        """Returns the nodes."""
        return self.nodes

    def get_edges(self) -> Iterator[Tuple[int, int, float]]:
        """Returns the edges."""
        if self.directed:
            for edge in self.edges:
                yield edge
        else:
            for (u,v,w) in self.edges:
                yield (u,v,w)
                yield (v,u,w)


class NetworkFromStateFile(Network):
    """
    A reader for state files.
    """
    def __init__(self, filename: str, directed):
        """
        Reads the file and stores nodes, state nodes, and edges in memory.

        Parameters
        ----------
        filename: str
            The state file to read.

        directed: bool = False
            Whether the network is directed.
        """
        self.directed   : bool                                  = directed
        self.nodes      : Dict[int, str]                        = dict()
        self.stateNodes : Dict[int, Dict[str, Union[int, str]]] = dict()
        self.edges      : List[Tuple[int, int, float]]          = list()

        with open(filename, "r") as fh:
            line = fh.readline()
            # skip comments
            while line.startswith("#"):
                line = fh.readline()

            # check for optional sections with nodes
            if line.startswith("*Vertices"):
                line = fh.readline()
                # skip comments
                while line.startswith("#"):
                    line = fh.readline()

                # read the nodes
                while not line.startswith("*States"):
                    nodeID, nodeLabel = line.strip().split()
                    self.nodes[int(nodeID)] = nodeLabel.strip("\"")
                    line = fh.readline()

            # now we must be at the start of the state nodes sections
            if not line.startswith("*States"):
                raise Exception(f"Expected *States nodes but got {line.strip()}")

            line = fh.readline()
            # skip comments
            while line.startswith("#"):
                line = fh.readline()

            # read the state nodes
            while not (line.startswith("*Edges") or line.startswith("*Links")):
                stateID, nodeID, stateLabel = splitQuotationAware(line.strip())
                self.stateNodes[int(stateID)] = dict( nodeID = int(nodeID)
                                                    , label  = stateLabel.strip("\"")
                                                    )
                line = fh.readline()

            line = fh.readline()
            # skip comments
            while line.startswith("#"):
                line = fh.readline()

            while line:
                edge = line.split()
                if len(edge) == 2:
                    u, v, w = int(edge[0]), int(edge[1]), 1.0
                elif len(edge) == 3:
                    u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
                else:
                    raise Exception(f"Unexpected edge format: {line.strip()}")
                self.edges.append((u,v,w))
                line = fh.readline()

    def get_nodes(self) -> Dict[int, str]:
        """Returns the nodes."""
        return self.nodes
    
    def get_state_nodes(self) -> Dict[int, Dict[str, Union[int, str]]]:
        """Returns the state nodes."""
        return self.stateNodes

    def get_edges(self) -> Iterator[Tuple[int, int, float]]:
        """Returns the edges."""
        if self.directed:
            for edge in self.edges:
                yield edge
        else:
            for (u,v,w) in self.edges:
                yield (u,v,w)
                yield (v,u,w)


class Partition(metaclass = ABCMeta):
    """
    Interface for partitions. Partitions can be read from files or directly
    from an Infomap instance.
    """
    def __init__(self) -> None:
        # all modules with their nodes
        self.modules : DefaultDict[Tuple[int, ...], Dict[str, Set[str] | float]] \
          = defaultdict(lambda: dict( nodes = set()
                                    , flow  = 0.0
                                    , enter = 0.0
                                    , exit  = 0.0
                                    )
                       )

        # flows of the nodes
        self.flows : Dict[str, float] = dict()

        # paths to nodes, such as "1:1:2" for node 2 in submodule 1 of module 1.
        self.paths : Dict[str, Tuple[int, ...]] = dict()

    def get_modules(self) -> DefaultDict[Tuple[int, ...], Dict[str, Set[int] | float]]:
        """
        Returns the modules.
        """
        return self.modules

    def get_flows(self) -> Dict[str, float]:
        """
        Returns the flows.
        """
        return self.flows

    def get_paths(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a dictionary for looking up the paths for nodes.
        """
        return self.paths


class PartitionFromTreeFile(Partition):
    """
    A class for reading partitions from tree files.
    """
    def __init__( self
                , treefile: str
                ) -> None:
        """
        Initialise and read the tree file.

        Parameters
        ----------
        treefile: str
            The tree file to read.
        """
        super().__init__()

        self.paths = dict()

        with open(treefile, "r") as fh:
            for line in fh:
                if line.startswith("*Links undirected"):
                    break

                if not line.startswith("#"):
                    path, flow, name, _nodeID = splitQuotationAware(line.strip())
                    path   = tuple([int(x) for x in path.split(":")])
                    self.flows[name] = float(flow)
                    self.paths[name] = path
                    for prefix in inits(self.paths[name]):
                        module = tuple(prefix)
                        self.modules[module]["nodes"].add(name)

        get_node_ID = id


class PartitionFromInfomap(Partition):
    """
    A class for reading partitions from Infomap instances using physical nodes.
    """
    def __init__( self
                , infomap: Infomap
                ) -> None:
        """
        Initialise and read the partition from the infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.

        netfile: Maybe[str] = none
            The file that contains the network (needed for of memory networks).
        """
        super().__init__()

        self.paths = dict()
        
        if len(infomap.names) > 0:
            self.node_IDs_to_labels : Dict[int, str] = infomap.names
            get_node_ID                              = lambda node: self.node_IDs_to_labels[node.node_id]
        else:
            get_node_ID                              = lambda node: node.node_id

        self._load_from_tree( tree        = infomap.get_tree()
                            , get_node_ID = get_node_ID
                            )
    
    def _load_from_tree( self
                       , tree
                       , get_node_ID: Callable[[int], str]
                       ) -> None:
        """
        Do the actual reading of the partition.

        Parameters
        ----------
        tree
            The tree from infomap.
        
        get_node_ID
            A function that takes and InfoNode and returns a node ID.
        """
        for node in tree:
            self.modules[node.path]["flow"]  = node.data.flow
            self.modules[node.path]["enter"] = node.data.enter_flow
            self.modules[node.path]["exit"]  = node.data.exit_flow

            if node.is_leaf:
                node_ID = get_node_ID(node)
                self.modules[node.path]["nodes"].add(node_ID)
                self.paths[node_ID] = node.path


class Module(metaclass = ABCMeta):
    def get_nodes(self) -> Set:
        raise NotImplemented
    
    def get_paths(self) -> List:
        raise NotImplemented


class Singleton(Module):
    def __init__(self, node, flow):
        self.nodes   = {node}
        self.flow    = flow
        self.address = ()
    
    def __repr__(self):
        return f"<Singleton: nodes={self.nodes}, flow={self.flow:.8f}>"
    
    def get_nodes(self) -> Set:
        return self.nodes
    
    def get_paths(self) -> List:
        path = ()
        info = dict( nodes = self.nodes
                , flow  = self.flow
                , enter = self.flow
                , exit  = self.flow
                )
        return [(path, info)]


class Sub(Module):
    def __init__(self, subs, partials, flow, enter, exit):
        self.subs     : List[Module] = subs
        self.partials : List[float]  = partials
        self.flow     : float        = flow
        self.enter    : float        = enter
        self.exit     : float        = exit
    
    def __repr__(self):
        return f"<Sub nodes={{{self.get_nodes()}}}, flow={self.flow}, enter={self.enter}, exit={self.exit}>"

    def get_nodes(self) -> Set:
        res = set()
        for partial, sub in zip(self.partials, self.subs):
            if partial > 0:
                for node in sub.get_nodes():
                    res.add(node)
        
        return res
    
    def get_paths(self) -> List:
        res          = []
        addr_counter = 0
        for partial, sub in zip(self.partials, self.subs):
            if partial > 0:
                addr_counter += 1
                for (path, info) in sub.get_paths():
                    path_ = (addr_counter,) + path
                    info_ = dict(nodes = info["nodes"], flow = partial * info["flow"], enter = partial * info["enter"], exit = partial * info["exit"])
                    res.append((path_, info_))
        return [((), dict(nodes = {}, flow = self.flow, enter = self.enter, exit = self.exit))] + res


class PartitionFromSoftAssignmentMatrices(Partition):

    """
    A class for converting a list of soft cluster assignment matrices to a partition.

    Because assignments are represented as a list of matrices, we assume that all
    modules have the same depth.
    """
    def __init__( self
                , S: List[np.array]
                , F: np.array
                , p: np.array
                , the_nodes: List
                ) -> None:
        """
        Initialise the partition and read it from the soft cluster assignment matrices.
        
        Parameters
        ----------
        S: List[np.array]
            A list of soft cluster assignment matrices. We assume that the first
            matrix represents the clustering at the highest level and the last one
            the assignment of nodes to modules at the bottom level.
        """
        assert len(S) > 0

        super().__init__()

        self.paths = defaultdict(set)

        # first, create singleton modules for the nodes
        node_singletons = []
        for node, flow in zip(the_nodes, p):
            node_singletons.append(Singleton(node = node, flow = flow))
        
        modules = node_singletons
        s       = S[0]

        C = F
        for s in S:
            C      = s.T @ C @ s
            diag_C = np.diag(C)
            q      = 1.0 - np.trace(C)
            q_m    = np.sum(C, axis = 1) - diag_C
            m_exit = np.sum(C, axis = 0) - diag_C
            p_m    = q_m + np.sum(C, axis = 0)

            modules_ = []
            for m, (flow, enter, exit) in enumerate(zip(p_m-q_m, q_m, m_exit)):
                modules_.append(Sub(subs = modules, partials = s[:,m], flow = flow, enter = enter, exit = exit))
            
            modules = modules_

        modules = Sub(subs = modules, partials = [1.0 for _ in modules], flow = 1.0, enter = 0.0, exit = 0.0)

        for path, module in modules.get_paths():
            if module["flow"] > 0 or len(module["nodes"]) > 0:
                self.modules[path]["flow"]  = module["flow"]
                self.modules[path]["enter"] = module["enter"]
                self.modules[path]["exit"]  = module["exit"]

                if len(module["nodes"]) > 0:
                    node_ID = list(module["nodes"])[0]
                    self.modules[path]["nodes"].add(node_ID)
                    self.paths[node_ID].add(path)