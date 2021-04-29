from typing import List, Tuple

from ..util import *

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
