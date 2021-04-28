from collections import defaultdict
import numpy as np
import sys

from typing import List, Optional, Tuple
from infomap import Infomap

def inits(l : List) -> List:
    """
    Generate the prefixes of a list, excluding the empty and full prefix!

    Parameters
    ----------
    l: List
        The list we want to get the inits of.

    """
    for init in range(1, len(l)):
        yield l[:init]


def splitQuotationAware(s : str, sep : Optional[str] = " ") -> List[str]:
    """
    Splits a string with an awareness of (non-nested!) double-quotation marks

    splitQuotationAware("1 \"Node one\"") will return ["1", "Node one"]
    as opposed to "1 \"Node one\"".split(" ") = ["1", "\"Node", "one\""]

    Parameters
    ----------
    s: str
        The string to split.

    sep: Optional[str] = " "
        The separator for splitting.

    """
    res = [""]
    i   = 0
    acc = ""
    while i < len(s):
        if s[i] == sep:
            res.append("")
        elif s[i] == "\"":
            i += 1
            while i < len(s) and s[i] != "\"":
                res[-1] += s[i]
                i += 1
        else:
            res[-1] += s[i]
        i += 1

    return res


class NetReaderFromDisk:
    """
    A reader for net files that contain only nodes and edges.
    We scan the net file for the start of nodes and edges so that we can jump
    there when requested instead of storing all nodes and edges in memory.

    This class is NOT thread-safe!
    """
    def __init__(self, filename : str):
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


class NetReader:
    """
    A reader for net files that contain only nodes and edges.
    """
    def __init__(self, filename: str):
        """
        Reads the file and stores nodes and edges in memory.

        Parameters
        ----------
        filename: str
            The net file to read.
        """
        self.nodes = []
        self.edges = []

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

    def get_edges(self) -> List[int]:
        """Returns the edges."""
        return self.edges


class StateFileReader:
    """
    A reader for state files.
    """
    def __init__(self, filename: str):
        """
        Reads the file and stores nodes, state nodes, and edges in memory.

        Parameters
        ----------
        filename: str
            The state file to read.
        """
        self.nodes      = list()
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
                    nodeID, _ = line.split()
                    self.nodes.append(int(nodeID))
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
                stateID, nodeID, _stateName = splitQuotationAware(line)
                self.stateNodes[int(stateID)] = int(nodeID)
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
        return self.edges


class CodeBook:
    """
    A code book to calculate path costs.
    A code book stores the flow, enter rate, and exit rate of a module as well
    as its sub-modules.
    """
    def __init__(self) -> None:
        """Creates an empty codebook."""
        self.code_book = dict()

    def __repr__(self) -> str:
        return "<CodeBook flow={:.2f}, enter={:.2f}, exit={:.2f}, norm={:.2f}, enter_cost={:.2f}, exit_cost={:.2f}, sub=[{:}]>".format(self.flow, self.enter, self.exit, self.normaliser, self.enter_cost, self.exit_cost, "|".join(["{:}={:}".format(k, v) for k, v in self.code_book.items()]))

    def insert_path(self, path: List[int], flow: float, enter: float, exit: float) -> None:
        """
        Inserts a path with corresponding flow data. A path can point to a module
        or a leaf node.

        Parameters
        ----------
        path: List[int]
            The path to a node that should be inserted.

        flow: float
            The flow of the node.

        enter: float
            The enter flow of the node.

        exit: float
            The exit flow of the node.

        Example
        -------
        ToDo
        """

        # We have reached the end of the path and insert the flows into the
        # current codebook.
        if len(path) == 0:
            self.flow  = flow
            self.enter = enter
            self.exit  = exit

        # We need to descend through the hierarchy of codebooks, clipping off
        # one piece of the path per step.
        else:
            if path[0] not in self.code_book:
                self.code_book[path[0]] = CodeBook()
            self.code_book[path[0]].insert_path(path[1:], flow, enter, exit)

    def calculate_normalisers(self) -> None:
        """
        Calculate the normalisation factors for all code books, that is the
        codebook usage rates.
        """
        self.normaliser = self.exit
        for m in self.code_book:
            self.normaliser += self.code_book[m].enter
            self.code_book[m].calculate_normalisers()

    def calculate_costs(self) -> None:
        """
        Calculate the enter and exit costs, in bits, for all codebooks.
        """
        self.exit_cost  = -np.log2(self.exit / self.normaliser) if self.normaliser > 0 and self.exit > 0 else 0.0
        self.enter_cost = 0.0

        for m in self.code_book:
            self.code_book[m].calculate_costs()
            self.code_book[m].enter_cost = -np.log2(self.code_book[m].enter / self.normaliser) if self.normaliser > 0 and self.code_book[m].enter > 0 else 0.0

    def get_flow(self, path: List[int]) -> float:
        """
        Returns the flow of the node that is addressed by `path`.

        Parameters
        ----------
        path: List[int]
            The path to the node whose flow we want to know.
        """
        if len(path) == 0:
            return self.flow
        return self.code_book[path[0]].get_flow(path[1:])

    def get_walk_cost(self, source: List[int], target: List[int]) -> float:
        """
        Returns the walk cost between two paths.
        For that, we first determine and clip off the common prefix of both paths.
        Then, we sum up the costs for exiting from the source path to the common
        ancestor and the costs for visiting the target path.

        Parameters
        ----------
        source: List[int]
            The source path, that is the address of a node.

        target: List[int]
            The target path, that is the address of a node.
        """
        if len(source) > 0 and len(target) > 0 and source[0] == target[0]:
            return self.code_book[source[0]].get_walk_cost(source[1:], target[1:])

        return self.get_path_cost_reverse(source) \
             + self.get_path_cost_forward(target)

    def get_path_cost_forward(self, path: List[int]) -> float:
        """
        Cost for descending along a path.

        Parameters
        ----------
        path: List[int]
            A path that addresses a node.
        """
        if len(path) == 1:
            return self.code_book[path[0]].enter_cost

        return self.code_book[path[0]].enter_cost + self.code_book[path[0]].get_path_cost_forward(path[1:])

    def get_path_cost_reverse(self, path: List[int]) -> float:
        """
        Cost for ascending along a path without visiting the starting node.

        Parameters
        ----------
        path: List[int]
            A path that addresses a node.
        """
        if len(path) == 1:
            return 0.0

        return self.code_book[path[0]].exit_cost + self.code_book[path[0]].get_path_cost_reverse(path[1:])


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
        self._load_modules(netfile, treefile)
        self._process_inputs()
        return self

    def load_from_infomap(self, infomap: Infomap) -> None:
        """
        Load a partition from an Infomap instance and construct codebooks from it.
        """
        raise Exception("PathCost:load_from_infomap not implemented.")

    def _process_inputs(self):
        """
        Private method for populating the codebooks after loading data.
        """

        # create the codebook and insert all paths
        self.cb = CodeBook()
        self.cb.insert_path((), 1, 0, 0)
        for m in self.modules:
            self.cb.insert_path( m.split(":")
                               , self.modules[m]["flow"]
                               , self.modules[m]["enter"]
                               , self.modules[m]["exit"]
                               )
        self.cb.calculate_normalisers()
        self.cb.calculate_costs()

    def _load_modules(self, netfile: str, treefile: str) -> None:
        """
        Load the modules.

        Parameters
        ----------
        netfile: str
            The file that contains the network.

        treefile: str
            The file that contains the partition.
        """
        # for reading the network
        reader  = StateFileReader(netfile)

        # all modules with their nodes
        modules = defaultdict(lambda: dict( nodes = set()
                                          , flow  = 0.0
                                          , enter = 0.0
                                          , exit  = 0.0
                                          )
                             )

        # node visit rates
        flows   = dict()

        # full paths of nodes
        paths   = dict()

        with open(treefile, "r") as fh:
            for line in fh:
                if line.startswith("*Links undirected"):
                    break

                if not line.startswith("#"):
                    path, flow, _, nodeID, _ = splitQuotationAware(line.strip())
                    nodeID = int(nodeID)
                    flows[nodeID] = float(flow)
                    paths[nodeID] = path.split(":")
                    for prefix in inits(paths[nodeID]):
                        module = ":".join(prefix)
                        modules[module]["nodes"].add(nodeID)

        # set the flow of modules by summing over the flow of contained nodes
        for n in reader.get_state_nodes():
            for i,_ in enumerate(paths[n], start=1):
                path = ":".join(paths[n][:i])
                modules[path]["flow"] += flows[n]

        # count all degrees for flow normalisation, assuming that the network is undirected
        degree = defaultdict(lambda: 0)
        for e in reader.get_edges():
            u,v,w = e
            degree[u] += w
            degree[v] += w

        # set enter and exit flow by going over all edges
        for e in reader.get_edges():
            u,v,w = e
            pfrom = paths[u]
            pto   = paths[v]

            i = 0
            while i < len(pfrom) and i < len(pto) and pfrom[i] == pto[i]:
                i += 1
            prefix = ":".join(pfrom[:i])

            pfrom = pfrom[i:]
            pto   = pto[i:]

            for j,_ in enumerate(pfrom, start=1):
                path = ":".join(pfrom[:j])
                if len(prefix) > 0:
                    path = prefix + ":" + path
                modules[path]["exit"]  += w/degree[u] * flows[u]
                modules[path]["enter"] += w/degree[v] * flows[v]

            for j,_ in enumerate(pto, start=1):
                path = ":".join(pto[:j])
                if len(prefix) > 0:
                    path = prefix + ":" + path
                modules[path]["exit"]  += w/degree[v] * flows[v]
                modules[path]["enter"] += w/degree[u] * flows[u]

        self.modules = modules
        self.paths   = paths

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
