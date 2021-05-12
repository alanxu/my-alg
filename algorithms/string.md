## Parsing

[722. Remove Comments](https://leetcode.com/problems/remove-comments/)

## Add Tag
[616. Add Bold Tag in String](https://leetcode.com/problems/add-bold-tag-in-string/)

```python
class Solution:
    def addBoldTag(self, s: str, dict: List[str]) -> str:
        S = len(s)
        paint = [False] * S
        
        for i in range(S):
            for word in dict:
                if s[i:].startswith(word):
                    # Trick: Batch assign value to sub array
                    paint[i:i + len(word)] = [True] * len(word)
        
        ans = []
        for i in range(S):
            if (i == 0 or not paint[i - 1]) and paint[i]:
                ans.append('<b>')
            ans.append(s[i])
            if (i == S - 1 or not paint[i + 1]) and paint[i]:
                ans.append('</b>')
            
        return ''.join(ans)
```

## Parenthsis

[678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)


## Performance Tuning

### [165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/)

## Shift Grouping

[249. Group Shifted Strings](https://leetcode.com/problems/group-shifted-strings/)

## Others

### [1328. Break a Palindrome](https://leetcode.com/problems/break-a-palindrome/)

```python
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        s = palindrome
        for i in range(len(s) // 2):
            if s[i] != 'a':
                # Trick: Replace a charactor in string
                return s[:i] + 'a' + s[i + 1:]
        return s[:-1] + 'b' if s[:-1] else ''
```


