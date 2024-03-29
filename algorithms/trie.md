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

### [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

```python
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        

    def addWord(self, word: str) -> None:
        node = self.trie
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = True
        

    def search(self, word: str) -> bool:
        
        def search_in_node(word, node):
            for i, c in enumerate(word):
                if c not in node:
                    if c == '.':
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]):
                                return True
                    return False
                node = node[c]
            return '$' in node
        
        return search_in_node(word, self.trie)
            
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```
```python
class TrieNode:
    def __init__(self):
        self.next = defaultdict(TrieNode)
        self.is_end = False

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.trie
        for x in word:
            cur = cur.next[x]
        cur.is_end = True

    def search(self, word: str) -> bool:
        # print(word)
        def _search(word, node):
            if not word:
                return node.is_end
            if word[0] == ".":
                for n in node.next.values():
                    if _search(word[1:], n):
                        return True
                return False
            else:
                node = node.next.get(word[0])
                if not node:
                    return 
                return _search(word[1:], node)
        
        return _search(word, self.trie)
```

### [212. Word Search II](https://leetcode.com/problems/word-search-ii/)

```python
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for w in word:
            node = node.children[w]
        node.is_word = True
    
    def search(self, word):
        node = self.root
        for w in word:
            node = node.children.get(w)
            if not node:
                return False
        return node.is_word
        
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        def dfs(r, c, path, node):
            if node.is_word:
                ans.append(path)
                node.is_word = False
            if r < 0 or r == rows or c < 0 or c == cols:
                return
            w = board[r][c]
            node = node.children.get(w)
            if not node:
                return
            board[r][c] = '#'
            dfs(r + 1, c, path + w, node)
            dfs(r - 1, c, path + w, node)
            dfs(r, c + 1, path + w, node)
            dfs(r, c - 1, path + w, node)
            board[r][c] = w
        
        ans, rows, cols = [], len(board), len(board[0])
        trie = Trie()
        for word in words:
            trie.insert(word)
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, "", trie.root)
        return ans
```


### [642. Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/)

```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False
        self.data = None
        self.rank = 0
    
class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        self.trie = TrieNode()
        self.keywords = ""
        for i, s in enumerate(sentences):
            self.add(s, times[i])

    def add(self, sentence, hot):
        cur = self.trie
        for c in sentence:
            cur = cur.children[c]
        cur.is_end = True
        cur.data = sentence
        cur.rank -= hot
    
    def dfs(self, node):
        ans = []
        if node:
            if node.is_end:
                ans.append((node.rank, node.data))
            for nxt in node.children.values():
                ans.extend(self.dfs(nxt))
        return ans
    
    def search(self, sentence):
        cur = self.trie
        for c in sentence:
            cur = cur.children[c]
        return self.dfs(cur)
        
    def input(self, c: str) -> List[str]:
        if c == '#':
            self.add(self.keywords, 1)
            self.keywords = ""
        else:
            self.keywords += c
            result = self.search(self.keywords)
            return [x[1] for x in sorted(result)[:3]]

# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)
```

### [1268. Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/)

```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False
        self.data = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        cur = self.root
        for w in word:
            cur = cur.children[w]

        cur.is_word = True
        cur.data = word
        
    def search(self, word):
        cur = self.root
        for w in word:
            cur = cur.children[w]
        return self.dfs(cur)
            
    def dfs(self, node):
        ans = []
        if node:
            if node.is_word:
                ans.append(node.data)
            for child in node.children.values():
                ans.extend(self.dfs(child))
        return ans
        
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        trie = Trie()
        for p in sorted(products):
            trie.insert(p)
        
        ans = []
        for i in range(1, len(searchWord) + 1):
            ans.append(trie.search(searchWord[:i])[:3])
        
        return ans
```