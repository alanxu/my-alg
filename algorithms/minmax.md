### [843. Guess the Word](https://leetcode.com/problems/guess-the-word/)

```python
# """
# This is Master's API interface.
# You should not implement it, or speculate about its implementation
# """
# class Master:
#     def guess(self, word: str) -> int:

class Solution:
    def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
        def match(w1, w2):
            return sum(c1 == c2 for c1, c2 in zip(w1, w2))
        
        x = 0
        while x < 6:
            match_count = collections.Counter(w1 for w1, w2 in itertools.permutations(wordlist, 2) if match(w1, w2) == 0)
            guess = min(wordlist, key=lambda x: match_count[x])
            x = master.guess(guess)
            wordlist = [w for w in wordlist if match(guess, w) == x]
```