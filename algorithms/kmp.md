# KMP
Time complexity: O(N). During the execution, j could be decreased at most N times and then increased at most N times, that makes overall execution time to be linear O(N).

### [28. Implement strStr()](https://leetcode.com/problems/implement-strstr/)

```python
# Fav
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # Alg: Sliding window
        if not needle:
            return 0
        left = right = 0
        N, K = len(haystack), len(needle)
        while left <= N - K:
            while left < N - K and haystack[left] != needle[0]:
                left += 1
                right = left
            for i in range(K):
                if haystack[right] == needle[i]:
                    if right - left + 1 == K:
                        return left
                    right += 1
                    
                else:
                    left += 1
                    right = left
                    break
        return -1
    
    def strStr(self, haystack: str, needle: str) -> int:
        # Alg: KMP
        # https://youtu.be/dgPabAsTFa8
        if not needle:
            return 0
        N, K = len(haystack), len(needle)
        lps = [-1] + [0] * K
        i, j = 0, -1
        while i < K:
            while j >= 0 and needle[i] != needle[j]:
                j = lps[j]
            i, j = i + 1, j + 1
            lps[i] = j
            
        i = j = 0
        while i < N:
            while j >= 0 and haystack[i] != needle[j]:
                j = lps[j]
            i, j = i + 1, j + 1
            if j == K:
                return i - K
        return -1
    
    def strStr(self, haystack: str, needle: str) -> int:
        # Alg: Rolling hash
        L, N = len(needle), len(haystack)
        if L > N:
            return -1

        a, modulus = 26, 2 ** 27 -1
        int_of = lambda c: ord(c) - ord('a')
        
        h_h = h_n = 0
        for i in range(L):
            h_h = (h_h * a + int_of(haystack[i])) % modulus
            h_n = (h_n * a + int_of(needle[i])) % modulus
        
        if h_h == h_n:
            return 0
        
        aL = pow(a, L, modulus)
        for start in range(1, N - L + 1):
            h_h = (h_h * a - int_of(haystack[start - 1]) * aL + int_of(haystack[start + L - 1])) % modulus
            if h_h == h_n:
                return start
            
        return -1
```

### [1392. Longest Happy Prefix](https://leetcode.com/problems/longest-happy-prefix/)

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        # KMP
        N = len(s)
        i, j = 0, -1
        lps = [-1] + [0] * N
        while i < N:
            while j >= 0 and s[i] != s[j]:
                j = lps[j]
            i, j = i + 1, j + 1
            lps[i] = j
        
        return s[:lps[-1]]
    
    def longestPrefix(self, s: str) -> str:
        # Rolling Hash
        N, a, modulus = len(s), 26, 2 ** 63 - 1
        h_prefix, h_suffix = 0, 0
        pos = -1
        for i in range(N - 1):
            cur_prefix, cur_suffix = s[i], s[N - i - 1]
            h_prefix = (a * h_prefix + (ord(cur_prefix) - ord('a'))) % modulus
            h_suffix = (pow(a, i, modulus) * (ord(cur_suffix) - ord('a')) + h_suffix) % modulus
            if h_prefix == h_suffix:
                pos = i
            
        return s[:pos + 1] if pos > -1 else ""
```

### [214. Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/)

```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        # Pattern: KMP - Longest prefix palindrome in string
        ss = s + '#' + s[::-1]
        
        N = len(ss)
        i, j = 0, -1
        lps = [-1] + [0] * N
        while i < N:
            while j >= 0 and ss[i] != ss[j]:
                j = lps[j]
            i, j = i + 1, j + 1
            lps[i] = j
        pos = lps[-1] - 1
        return s[pos + 1:][::-1] + s
```

### [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)

```python
# Fav
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        # Intuition: KMP - Get longest common prefix/suffix
        # at the end of s. The len of repeated pattern is N - l.
        # if l == 0, there is no repeated pattern. Also N - l should
        # be divisor of N cuz the whole s compose of repeated pattern
        N = len(s)
        lps = [-1] + [0] * N
        i, j = 0, -1
        while i < N:
            while j >= 0 and s[i] != s[j]:
                j = lps[j]
            i, j = i + 1, j + 1
            lps[i] = j
            
        l = lps[-1]
        return l != 0 and N % (N - l) == 0
    
    def repeatedSubstringPattern(self, s: str) -> bool:
        import re
        pattern = re.compile(r"^(.+)\1+$")
        return pattern.match(s)
    
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s + s)[1:-1]
```

### [1764. Form Array by Concatenating Subarrays of Another Array](https://leetcode.com/problems/form-array-by-concatenating-subarrays-of-another-array/)

```python
class Solution:
    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        # Intuition: KMP for array
        # Sligth modification to kmp() adding start index for source array,
        # the i second loop start from 'start' rather than 0. Then loop
        # the groups, every kmp search start from index after prev group
        def kmp(s, t, start):
            M, N = len(s), len(t)
            i, j = 0, -1
            lps = [-1] + [0] * N
            while i < N:
                while j >= 0 and t[i] != t[j]:
                    j = lps[j]
                i, j = i + 1, j + 1
                lps[i] = j
                
            i, j = start, 0
            while i < M:
                while j >= 0 and s[i] != t[j]:
                    j = lps[j]
                i, j = i + 1, j + 1
                if j == N:
                    return i - N
            return -1
        
        pos, l = 0, 0
        for i in range(len(groups)):
            pos, l = kmp(nums, groups[i], pos + l), len(groups[i])
            if pos < 0:
                return False

        return True
```