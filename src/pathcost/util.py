from typing import List, Optional

def inits(l: List) -> List:
    """
    Generate the prefixes of a list, excluding the empty and full prefix!

    Parameters
    ----------
    l: List
        The list we want to get the inits of.

    """
    for init in range(1, len(l)):
        yield l[:init]


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
