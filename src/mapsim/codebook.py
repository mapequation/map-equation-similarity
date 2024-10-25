from __future__  import annotations
from dataclasses import dataclass, field
from heapq       import heappush, heappop
from numpy       import log2, inf
from typing      import Any, Dict, List, Optional as Maybe, Tuple
from scipy.stats import entropy


@dataclass(order=True)
class PrioritisedItem:
    cost: int
    item: Any=field(compare=False)


def assignCodewords(t : Tuple, prefix : str = "") -> List[Tuple[Any, str]]:
    """
    Assigns codewords to the tree represented by the given tuple t.

    Parameters
    ----------
    t : Tuple
        The Huffman tree.

    prefix : str = ""
        The prefix that should be prepended to all codewords.

    Returns
    -------
    List[Tuple[Any, str]]
        A list of symbols and their assigned codewords.
    """
    if type(t) == tuple:
        l,r = t
        return assignCodewords(l, prefix = prefix + "0") \
             + assignCodewords(r, prefix = prefix + "1")

    else:
        return [(t, prefix)]
        

def mkHuffmanCode(X : List[Any], P : List[float]) -> Dict[Any, str]:
    """
    Creates a Huffman code for the given symbols with probabilities P.

    Parameters
    ----------
    X : List[Any]
        a list of symbols

    P : List[float]
        the symbols' probabilities

    Returns
    -------
    Dict[Any, str]
        A dictionary with symbols as keys and codewords as values.
    """
    q = []

    # use a head to keep the partial trees sorted
    for (x, p) in zip(X, P):
        heappush(q, PrioritisedItem(cost = p, item = x))

    while len(q) > 1:
        l = heappop(q)
        r = heappop(q)
        heappush(q, PrioritisedItem(cost = l.cost + r.cost, item = (l.item, r.item)))

    t = heappop(q).item
    return dict(assignCodewords(t))


class CodeBook:
    """
    A code book to calculate path costs.
    A code book stores the flow, enter rate, and exit rate of a module as well
    as its sub-modules.
    """
    def __init__(self) -> None:
        """Creates an empty codebook."""
        self.node       : Maybe[str]          = None
        self.code_book  : Dict[int, CodeBook] = dict()
        self.code_words : Dict[int, str]      = None
        self.flow       : float               = 0.0
        self.enter      : float               = 0.0
        self.exit       : float               = 0.0
        self.normaliser : float               = 0.0


    def __repr__(self) -> str:
        """
        Serialises the codebook.

        Returns
        -------
        str
            The serialised codebook.
        """
        return f"<CodeBook\n{self._serialise(indent = 0)}\n>"


    def _serialise(self, indent: int, codeword = "") -> str:
        """
        Serialises the codebook in a somewhat readable format.

        Parameters
        ----------
        indent: int
            The indentation level

        Returns
        -------
        str"
            The serialised codebook
        """
        subs  = "".join([f"\n{cb._serialise(indent = indent + 4, codeword = self.code_words[k])}" for k,cb in self.code_book.items()])
        
        if self.node is not None:
            return indent * " " + f"node={self.node}, flow={self.flow:.2f}, visit_cost={self.enter_cost:.2f}, codeword={codeword}, {subs}"
        else:
            extra = ""
            if self.exit > 0:
                extra = f"exit={self.code_words['exit']},"

            return indent * " " + f"flow={self.flow:.2f}, enter={self.enter:.2f}, exit={self.exit:.2f}, norm={self.normaliser:.2f}, enter_cost={self.enter_cost:.2f}, exit_cost={self.exit_cost:.2f}, codeword={codeword}, {extra} {subs}"


    def mk_codewords(self) -> None:
        """
        Traverses the codebook and assigns codewords to all modules and nodes.
        """
        self.code_words = dict()

        X = list(self.code_book.keys())
        P = [self.code_book[x].flow for x in X]

        if self.exit > 0:
            X += ["exit"]
            P += [self.exit]

        if len(X) > 1:
            self.code_words = mkHuffmanCode(X = X, P = P)
        else:
            self.code_words = dict()

        for cb in self.code_book.values():
            cb.mk_codewords()


    def _get_address_node_links(self) -> List:
        """
        Traverses the codebook and returns the links in the coding tree, using
        the modules' addresses as node names.
        """
        res = []

        for k,v in self.code_book.items():
            # if there's only a node in the codebook, we've reached the leaf
            if v.node is not None:
                res += [f"{k}"]

            # otherwise, we traverse the sub-modules
            else:
                res += [(k,f"{k}:{l}") for l in v._get_address_node_links()]

        return res


    def get_nodes(self) -> List[str]:
        """
        Traverses the codebook and collects the nodes.
        """
        if self.node is not None:
            return [self.node]
        
        else:
            res = []

            for cb in self.code_book.values():
                res += cb.get_nodes()
            
            return res
    
    def get_module(self, path: Tuple[int, ...]) -> CodeBook:
        """
        Retrieves the submodule at the given path.

        Parameters
        ----------
        path: List[int]
            The path to the submodule of interest.

        Returns
        -------
        CodeBook
            The codebook at the given path.
        """
        if path == ():
            return self

        return self.code_book[path[0]].get_module(path[1:])


    def insert_path(self, node: str, path: Tuple[int, ...], flow: float, enter: float, exit: float) -> None:
        """
        Inserts a path with corresponding flow data. A path can point to a module
        or a leaf node.

        Parameters
        ----------
        node: str
            The node we are inserting.

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
            self.node  : Maybe[str] = node
            self.flow  : float      = flow
            self.enter : float      = enter
            self.exit  : float      = exit

        # We need to descend through the hierarchy of codebooks, clipping off
        # one piece of the path per step.
        else:
            if path[0] not in self.code_book:
                self.code_book[path[0]] = CodeBook()
            self.code_book[path[0]].insert_path( node  = node
                                               , path  = path[1:]
                                               , flow  = flow
                                               , enter = enter
                                               , exit  = exit
                                               )


    def calculate_normalisers(self) -> None:
        """
        Calculate the normalisation factors for all codebooks, that is the
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
        self.exit_cost: float  = -log2(self.exit / self.normaliser) if self.normaliser > 0.0 and self.exit > 0.0 else inf
        self.enter_cost = 0.0

        for m in self.code_book:
            self.code_book[m].calculate_costs()
            self.code_book[m].enter_cost = -log2(self.code_book[m].enter / self.normaliser) if self.normaliser > 0.0 and self.code_book[m].enter > 0.0 else inf


    def get_flow(self, path: Tuple[int, ...]) -> float:
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


    def get_walk_rate(self, source: Tuple[int, ...], target: Tuple[int, ...]) -> float:
        """
        Rate of walking along the path from `source` to `target`.
        For that, we first determine and clip off the common prefix of both paths.
        Then, we sum up the costs for exiting from the source path to the common
        ancestor and the costs for visiting the target path.

        Parameters
        ----------
        source: List[int]
            The source path, that is the address of a node.

        target: List[int]
            The target path, that is the address of a node.

        Returns
        -------
        float
            The rate of walking along the path between the given `source` and `target`.
        """
        # self-link
        if len(source) == 1 and len(target) == 1 and source[0] == target[0]:
            return (self.code_book[target[0]].enter / self.normaliser) if self.normaliser > 0.0 else 0.0

        # clip off common prefixes
        if len(source) > 0 and len(target) > 0 and source[0] == target[0]:
            return self.code_book[source[0]].get_walk_rate(source[1:], target[1:])

        return self.get_path_rate_reverse(source) \
             * self.get_path_rate_forward(target)


    def get_path_rate_forward(self, path: Tuple[int, ...]) -> float:
        """
        Rate of descending along a path.

        Parameters
        ----------
        path: List[int]
            A path that addresses a node.

        Returns
        -------
        float
            The rate of descending along the given path.
        """
        if len(path) == 1:
            return (self.code_book[path[0]].enter / self.normaliser) if self.normaliser > 0.0 else 0.0

        return ( self.code_book[path[0]].enter / self.normaliser
               * self.code_book[path[0]].get_path_rate_forward(path[1:])
               ) if self.normaliser > 0.0 else 0.0


    def get_path_rate_reverse(self, path: Tuple[int, ...]) -> float:
        """
        The rate of ascending along a path without visiting the
        starting node.

        Parameters
        ----------
        path: List[int]
            A path that addresses a node.

        Returns
        -------
        float
            The rate of ascending along the given path.
        """
        if len(path) == 1:
            return 1.0

        return ( self.code_book[path[0]].exit / self.code_book[path[0]].normaliser
               * self.code_book[path[0]].get_path_rate_reverse(path[1:])
               ) if self.code_book[path[0]].normaliser > 0.0 else 0.0


    def traverse_for_recommendations(self, entry_cost : float, targets : List[PrioritisedItem]):
        # there's only this one node to visit
        if self.node is not None:
            yield self.node
            return targets

        for k in self.code_book.keys():
            heappush(targets, PrioritisedItem(cost = entry_cost - log2(self.code_book[k].enter / self.normaliser), item = self.code_book[k]))
        return targets


    def recommend(self, path : Tuple[int, ...]):
        # recurse to the module where the start node is located
        if len(path) == 1:
            return_cost, targets = 0, []
        else:
            return_cost, targets = yield from self.code_book[path[0]].recommend(path[1:])

        exit_cost = - log2(self.exit / self.normaliser) if self.exit > 0.0 and self.normaliser > 0.0 else inf
        for k in self.code_book.keys():
            # if we're in the start module, use all sub-modules as targets
            # otherwise ignore the module from which we came back from the recursion
            if len(path) == 1 or k != path[0]:
                heappush(targets, PrioritisedItem(cost = return_cost - log2(self.code_book[k].enter / self.normaliser), item = self.code_book[k]))

        while len(targets) > 0:
            target : PrioritisedItem = heappop(targets)
            if target.cost < exit_cost:
                targets = yield from target.item.traverse_for_recommendations(entry_cost = target.cost, targets = targets)
            else:
                heappush(targets, target)
                break

        return exit_cost, targets


    def divergence(self : CodeBook, Q : CodeBook, source: Tuple[Tuple[int, ...]], targets: List[Tuple[Tuple[int, ...]]]) -> float:
        P    = self
        D_KL = 0.0

        sourceP, sourceQ = source

        rPs = []
        rQs = []

        for target in targets:
            targetP, targetQ = target

            rPs.append(P.get_walk_rate(source = sourceP, target = targetP))
            rQs.append(Q.get_walk_rate(source = sourceQ, target = targetQ))

        rPs = [rP/sum(rPs) for rP in rPs]
        rQs = [rQ/sum(rQs) for rQ in rQs]

        for rP, rQ in zip(rPs, rQs):
            D_KL += (rP/sum(rPs)) * log2((rP/sum(rPs)) / (rQ/sum(rQs)))

        return P.get_flow(path = sourceP) * D_KL


    def L(self : CodeBook) -> float:
        if self.node is not None:
            return self.enter

        Q = [m.enter for m in self.code_book.values()] + [self.exit]

        return sum(Q) * entropy(Q, base = 2) \
             + sum([m.flow * m.L() for m in self.code_book.values()])