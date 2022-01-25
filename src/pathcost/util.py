from typing import Any, Iterator, List, Optional

def inits(l: List[Any]) -> Iterator[List[Any]]:
    """
    Generate the prefixes of a list, excluding the empty and full prefix!

    Parameters
    ----------
    l: List
        The list we want to get the inits of.

    """
    for init in range(1, len(l)):
        yield l[:init]


def suffixes(l: List[Any]) -> Iterator[List[Any]]:
    """
    Generates the suffixes of a list, including the empty and full suffix!

    Parameters
    ----------
    l: List
        The list we want to get the tails of.
    """
    for suffix in range(len(l) + 1):
        yield l[suffix:]


def isPrefix(l1: List[Any], l2: List[Any]) -> bool:
    """
    Checks whether l1 is a prefix of l2.
    For example, [1,2] is a prefix of [1,2,3,4] but [1,3] is not.

    Parameters
    ----------
    l1: List
        The first list.

    l2: List
        The second list.
    """
    if l1 == []:
        return True
    
    return len(l1) <= len(l2) \
       and l1[0] == l2[0] \
       and isPrefix(l1[1:], l2[1:])


def splitQuotationAware(s: str, sep: Optional[str] = " ") -> List[str]:
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
