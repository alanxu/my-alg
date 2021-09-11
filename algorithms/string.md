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

### [9. Palindrome Number](https://leetcode.com/problems/palindrome-number/)

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        x = str(x)
        return x == x[::-1]
```

### [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        """
        Intuition: Two pointer to compare left and right and
        move both left/right toward centrer.
        When s[l] != s[r], try to skip l or r, and
        check the remaining substr.
        The alg is generic by supporting multiple deletes.
        Use recursion to handle that.
        """
        def validate(left, right, deletes):
            if deletes == 0:
                _s = s[left:right+1]
                return _s == _s[::-1]
            l, r = left, right
            # If len(s) is even, compare every pair, if odd,
            # mid is not compared and no need to compare
            while l < r:
                if s[l] != s[r]:
                    return validate(l, r - 1, deletes - 1) or\
                        validate(l + 1, r, deletes - 1)
                l += 1
                r -= 1
            return True
        return validate(0, len(s) - 1, 1)
```

### [68. Text Justification](https://leetcode.com/problems/text-justification/)

### [921. Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/)

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        ans = bal = 0
        for x in s:
            if x == '(':
                bal += 1
            else:
                bal -= 1
            if bal == -1:
                ans += 1
                bal += 1
        return ans + bal
```