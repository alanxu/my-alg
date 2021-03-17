### [336. Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/)

```python
# [ "A", "B", "BAN", "BANANA", "BAT", "LOLCAT", "MANA", "NAB", "NANA", "NOON", "ON", "TA", "TAC"]
# [ "A", "B", "NAB", "ANANAB", "TAB", "TACLOL", "ANAM", "BAN", "ANAN", "NOON", "NO", "AT", "CAT"]
# Case 1: ["A" "A"], ["A" "A"]
# Case 2: ["TAC" "LOLCAT"], ["TAC" "TACLOL"]
# Case 3: ["BANANA" "B"], ["BANANA" "B"]
# The order of two string matters

# Trie's root is always empty

class TrieNode:
    def __init__(self):
        self.next = defaultdict(TrieNode)
        self.ending_word = -1
        self.palindrome_suffixes = []

class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        trie = TrieNode()
        for i, word in enumerate(words):
            word = word[::-1]
            cur = trie
            for j, c in enumerate(word):
                # Do this before current node (j, c), that is before call cur = cur.next[c]
                # because you want to set the node before corrent (j, c) node if cur node is
                # palindrome
                # Trick: Check palindrom from j
                if word[j:] == word[j:][::-1]:
                    cur.palindrome_suffixes.append(i)
                cur = cur.next[c]
                
            cur.ending_word = i
        
        ans = []
        for i, word in enumerate(words):
            cur = trie
            for j, c in enumerate(word):
                # Do this before current node (j, c), that is before call cur = cur.next[c]
                # because you need to handle edge case word is 'a' and '' is at root of trie with
                # ending_word != 1. When meet end of word, check if [current pos:] is palindrome
                if cur.ending_word != -1:
                    if word[j:] == word[j:][::-1]:
                        ans.append([i, cur.ending_word])
                
                # This check should be performed after prev one, c need not to exist in order to 
                # do pre check
                if c not in cur.next:
                    break

                cur = cur.next[c]
            # Trick: For/Else, else is executed if no break
            else:
                # If it is here, it means the searching word matched a word completely or matched part of a word
                # the for loop ends without break, it meas the word is inerated to the end, it might or might not
                # hit the end of that branch of trie
                # If it is also end of word, it is perfect match, also it cannot be itself
                if cur.ending_word != -1 and cur.ending_word != i:
                    ans.append([i, cur.ending_word])
                # On whereever the word stops, check if there is any palindrom starting from next of cur, if 
                # yes count it
                for j in cur.palindrome_suffixes:
                    ans.append([i, j])
        return ans
```

### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

```python
class TrieNode:
    def __init__(self):
        self.next = defaultdict(TrieNode)
        self.is_end = False

class Trie:

    def __init__(self):
        self.trie = TrieNode()
        
    def insert(self, word: str) -> None:
        cur = self.trie
        for c in word:
            cur = cur.next[c]
        cur.is_end = True
        
    def search(self, word: str) -> bool:
        cur = self.trie
        for c in word:
            if c not in cur.next:
                return False
            cur = cur.next[c]
        return cur.is_end
        
    def startsWith(self, prefix: str) -> bool:
        cur = self.trie
        for c in prefix:
            if c not in cur.next:
                return False
            cur = cur.next[c]
        return True
```