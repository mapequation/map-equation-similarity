from numpy  import log2, inf
from typing import List

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
        """
        """
        return f"<CodeBook\n{self._serialise(indent = 0)}\n>"

    def _serialise(self, indent: int) -> str:
        """
        """
        subs = "".join([f"\n{cb._serialise(indent + 4)}" for cb in self.code_book.values()])
        return indent * " " + f"flow={self.flow:.2f}, enter={self.enter:.2f}, exit={self.exit:.2f}, norm={self.normaliser:.2f}, enter_cost={self.enter_cost:.2f}, exit_cost={self.exit_cost:.2f} {subs}"


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
        self.exit_cost  = -log2(self.exit / self.normaliser) if self.normaliser > 0.0 and self.exit > 0.0 else inf
        self.enter_cost = 0.0

        for m in self.code_book:
            self.code_book[m].calculate_costs()
            self.code_book[m].enter_cost = -log2(self.code_book[m].enter / self.normaliser) if self.normaliser > 0.0 and self.code_book[m].enter > 0.0 else inf

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

        return self.code_book[path[0]].enter_cost \
             + self.code_book[path[0]].get_path_cost_forward(path[1:])

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

        return self.code_book[path[0]].exit_cost \
             + self.code_book[path[0]].get_path_cost_reverse(path[1:])
