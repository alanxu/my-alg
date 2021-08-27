# Multi Patter Search
Multiple pattern search in a string. All such problems usually could be solved by sliding window approach in 
a linear time. The challenge here is how to implement constant-time slice to fit into this linear time.

If the patterns are not known in advance, i.e. it's "find duplicates" problem, one could use one of two ways 
to implement constant-time slice: Bitmasks or Rabin-Karp. Please check article Repeated DNA Sequences 
for the detailed comparison of these two algorithms.


## KMP

### [28. Implement strStr()](https://leetcode.com/problems/implement-strstr/)

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
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

## Rolling Hash
### [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)
### [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)
### [796. Rotate String](https://leetcode.com/problems/rotate-string/)

### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        Ls, Lp = len(s), len(p)
        
        if Lp > Ls:
            return []
        
        p_count, s_count = [0] * 26, [0] * 26
        for ch in p:
            p_count[ord(ch) - ord('a')] += 1
        
        ans = []
        
        # Trick: Sliding window
        for right in range(Ls):
            s_count[ord(s[right]) - ord('a')] += 1
            
            # When window is longer than Lp
            if right >= Lp:
                # 
                s_count[ord(s[right - Lp]) - ord('a')] -= 1
                
            if p_count == s_count:
                ans.append(right - Lp + 1)
            
            
        return ans
```

### [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        def search(l):
            hash_ = 0
            for i in range(l):
                hash_ = (hash_ * A + nums[i]) % MOD
                
            seen = {hash_}
            
            # Because origin start will be reoved after multiplied by
            # one more A, so power by l rather l - 1.
            aL = pow(A, l, MOD)
            for start in range(1, N - l + 1):
                hash_ = (hash_ * A - nums[start - 1] * aL + nums[start + l - 1]) % MOD
                if hash_ in seen:
                    print(start)
                    return start
                seen.add(hash_)
            return -1

        nums = [ord(c) - ord('a') for c in s]
        A, MOD = 26, 2**63 - 1
        N = len(s)
        
        left, right = 0, N
        while left <= right:
            l = (left + right) // 2
            if search(l) != -1:
                left = l + 1
            else:
                right = l - 1
        # print(left)
        start = search(left - 1)
        return s[start: start + left - 1]
```

### [1062. Longest Repeating Substring](https://leetcode.com/problems/longest-repeating-substring/)

```python
class Solution:
    def longestRepeatingSubstring(self, S: str) -> int:
        def search(L, a, modulus, N, nums):
            """
            Alg: Rabin-Karp with polynomial rolling hash.
            Search a substring of given length
            that occurs at least 2 times.
            @return start position if the substring exits and -1 otherwise.
            """
            h = 0
            for i in range(L):
                h = (h * a + nums[i]) % modulus
            
            seen = {h}
            
            # Because origin start will be reoved after multiplied by
            # one more A, so power by l rather l - 1.
            aL = pow(a, L, modulus)
            
            for start in range(1, N - L + 1):
                h = (h * a - nums[start - 1] * aL + nums[start + L - 1])
                h %= modulus
                if h in seen:
                    return start
                seen.add(h)
            
            return -1
        
        N = len(S)
        nums = [ord(x) - ord('a') for x in S]
        a, modulus = 26, 2**24
        lo, hi = 0, len(S) - 1
        # We have to use template 1 here, because when
        # there is a match for a length, we need to increase
        # the length and continue to search, so we can not
        # put hi = mi on match condition, instead it has to be
        # lo = mi + 1, then it has to be template 1.
        while lo <= hi:
            mi = lo + (hi - lo) // 2
            if search(mi, a, modulus, N, nums) != -1:
                lo = mi + 1
            else:
                hi = mi - 1
                
        return hi
```

### [1554. Strings Differ by One Character](https://leetcode.com/problems/strings-differ-by-one-character/)

```python
class Solution:
    def differByOne(self, dict: List[str]) -> bool:
        M, N = len(dict[0]), len(dict)
        # a was working at 26...
        a, modulus = 261, 10 ** 9 + 7
        hashh = [0] * N
        for i in range(N):
            # Trick: Caclculate hash for a str
            h = 0
            # Start from 0, idx 0 will be the most significant
            # cuz it multiply by a the most times
            for j in range(M):
                h = (h * a + ord(dict[i][j]) - ord('a'))
                h %= modulus
            hashh[i] = h
        
        K = 1
        # Start from M - 1 which is the lowest position
        for j in range(M - 1, -1, -1):
            seen = set()
            for i in range(N):
                h = hashh[i] + modulus - (ord(dict[i][j]) - ord('a')) * K % modulus
                h %= modulus
                if h in seen:
                    return True
                seen.add(h)
            K = K * a % modulus
        
        return False
```