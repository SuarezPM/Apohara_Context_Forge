"""PrefixDedup — lightweight token-count + shared-prefix helper.

The CompressionCoordinator needs two cheap operations on raw context
strings: count a context's tokens, and find the longest shared prefix
between two contexts. This is the production default for
``ContextRegistry.dedup``; tests inject their own fake with the same
contract (``count_prefix_tokens``, ``find_shared_prefix``).
"""

from typing import Optional

from apohara_context_forge.token_counter import TokenCounter


class PrefixDedup:
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._tc = token_counter or TokenCounter.get()

    def count_prefix_tokens(self, prefix: str) -> int:
        return self._tc.count(prefix)

    def find_shared_prefix(self, a: str, b: str) -> str:
        low = 0
        high = min(len(a), len(b))
        while low < high:
            mid = (low + high + 1) // 2
            if a.startswith(b[:mid]):
                low = mid
            else:
                high = mid - 1
        i = low
        if i == min(len(a), len(b)):
            return a[:i]
        # Back off to the last word boundary so we don't split a token.
        j = a.rfind(" ", 0, i)
        return a[:j] if j > 0 else a[:i]
