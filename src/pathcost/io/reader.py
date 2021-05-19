from abc         import ABCMeta, abstractmethod
from collections import defaultdict
from infomap     import Infomap
from math        import trunc
from typing      import Dict, List, Tuple

from ..util import *

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
        pass


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
        self.directed = directed
        self.nodes    = []
        self.edges    = []

        with open(filename, "r") as fh:
            line = fh.readline()
            while not line.startswith("*Vertices"):
                line = fh.readline()

            line = fh.readline()
            while not (line.startswith("*Edges") or line.startswith("*Links")):
                n, _ = line.split()
                self.nodes.append(int(n))
                line = fh.readline()

            while line:
                edge = line.split()
                if len(edge) == 2:
                    u, v, w = edge[0], edge[1], 1.0
                elif len(...) == 3:
                    u, v, w = edge[0], edge[1], edge[2]
                self.edges.append((int(u),int(v),float(w)))
                line = fh.readline()

    def get_nodes(self) -> List[int]:
        """Returns the nodes."""
        return self.nodes

    def get_edges(self) -> List[Tuple[int, int, float]]:
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
        self.directed   = directed
        self.nodes      = dict()
        self.stateNodes = dict()
        self.edges      = list()

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

    def get_nodes(self) -> List[int]:
        """Returns the nodes."""
        return self.nodes
    
    def get_state_nodes(self) -> List[int]:
        """Returns the state nodes."""
        return self.stateNodes

    def get_edges(self) -> List[Tuple[int, int, float]]:
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
    def __init__(self):
        # all modules with their nodes
        self.modules = defaultdict(lambda: dict( nodes = set()
                                               , flow  = 0.0
                                               , enter = 0.0
                                               , exit  = 0.0
                                               )
                                  )

        # flows of the nodes
        self.flows = dict()

        # paths to nodes, such as "1:1:2" for node 2 in submodule 1 of module 1.
        self.paths = None

    def get_modules(self) -> Dict:
        """
        Returns the modules.
        """
        return self.modules

    def get_flows(self) -> Dict:
        """
        Returns the flows.
        """
        return self.flows

    def get_paths(self) -> Dict:
        """
        Returns a dictionary for looking up the paths for nodes.
        """
        return self.paths


class PartitionFromTreeFile(Partition):
    """
    A class for reading partitions from tree files.
    """
    def __init__(self, treefile: str) -> None:
        """
        Initialise and read the tree file.

        Parameters
        ----------
        treefile: str
            The tree file to read.
        """
        super().__init__()

        with open(treefile, "r") as fh:
            for line in fh:
                if line.startswith("*Links undirected"):
                    break

                if not line.startswith("#"):
                    path, flow, _, nodeID, _ = splitQuotationAware(line.strip())
                    path   = tuple([int(x) for x in path.split(":")])
                    nodeID = int(nodeID)
                    self.flows[nodeID] = float(flow)
                    self.paths[nodeID] = path
                    for prefix in inits(self.paths[nodeID]):
                        module = tuple(prefix)
                        self.modules[module]["nodes"].add(nodeID)


class PartitionFromInfomap(Partition):
    """
    A class for reading partitions from Infomap instances using physical nodes.
    """
    def __init__(self, infomap: Infomap) -> None:
        """
        Initialise and read the partition from the infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
        """
        super().__init__()

        if infomap.memoryInput:
            self.paths = defaultdict(lambda: dict())
        else:
            self.paths = dict()

        self.node_IDs_to_labels = infomap.names

        def state_ID_to_memory_label(state_ID, num_nodes = len(infomap.names)):
            memory = trunc((state_ID-2)/(num_nodes-1))
            return self.node_IDs_to_labels[memory] if memory > 0 else "{}"

        self.state_ID_to_memory_label = state_ID_to_memory_label

        self._load_from_tree( tree        = infomap.get_tree()
                            , get_node_ID = lambda node: self.node_IDs_to_labels[node.node_id]
                            , with_state  = infomap.memoryInput
                            )
    
    def _load_from_tree(self, tree, get_node_ID, with_state: bool) -> None:
        """
        Do the actual reading of the partition.

        Parameters
        ----------
        tree
            The tree from infomap.
        
        get_node_ID
            A function that takes and InfoNode and returns a node ID.

        with_state: bool
            Whether the partition is for a state network.
        """
        for node in tree:
            self.modules[node.path]["flow"]  = node.data.flow
            self.modules[node.path]["enter"] = node.data.enter_flow
            self.modules[node.path]["exit"]  = node.data.exit_flow

            if node.is_leaf:
                node_ID = get_node_ID(node)
                self.modules[node.path]["nodes"].add(node_ID)

                if with_state:
                    self.paths[node_ID][self.state_ID_to_memory_label(state_ID = node.state_id)] = node.path
                else:
                    self.paths[node_ID] = node.path


class StatePartitionFromInfomap(Partition):
    """
    A class for reading partitions from Infomap instances using state nodes.
    """
    def __init__(self, infomap: Infomap) -> None:
        """
        Initialise and read the partition from the infomap instance.

        Parameters
        ----------
        infomap: Infomap
            The infomap instance.
        """
        super().__init__()

        self._load_from_tree( tree        = infomap.tree
                            , get_node_ID = lambda node: node.state_id
                            )