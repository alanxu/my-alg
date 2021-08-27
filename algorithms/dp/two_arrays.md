
### [97. Interleaving String](https://leetcode.com/problems/interleaving-string/)

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # Pattern: DP type2
        M, N, O = len(s1), len(s2), len(s3)
        if M + N != O: return False
        dp = [[False] * (N + 1) for _ in range(M + 1)]
        dp[0][0] = True
        
        for i in range(1, M + 1):
            if s1[:i] == s3[:i]:
                dp[i][0] = True
    
        for j in range(1, N + 1):
            if s2[:j] == s3[:j]:
                dp[0][j] = True
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if s1[i - 1] == s3[i + j - 1] and dp[i - 1][j] or \
                   s2[j - 1] == s3[i + j - 1] and dp[i][j - 1]:
                    dp[i][j] = True
                    
        return dp[-1][-1]
    
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        M, N, O = len(s1), len(s2), len(s3)
        if M + N != O: return False
        dp = [[False] * (N + 1) for _ in range(M + 1)]
        
        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    dp[i][j] = True
                elif i == 0:
                    if s2[:j] == s3[:j]: dp[i][j] = True
                elif j == 0:
                    if s1[:i] == s3[:i]: dp[i][j] = True
                elif s1[i - 1] == s3[i + j - 1] and dp[i - 1][j] or \
                   s2[j - 1] == s3[i + j - 1] and dp[i][j - 1]:
                    dp[i][j] = True
                    
        return dp[-1][-1]
    
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        M, N, O = len(s1), len(s2), len(s3)
        if M + N != O: return False
        dp = [False] * (N + 1)
        # Trick: State compression
        # dp[j] = s1[i - 1] == s3[i + j - 1] and dp[j] or \
        #         s2[j - 1] == s3[i + j - 1] and dp[j - 1]
        # When check dp[j] at right of =, it is prev value
        # Trick: When convert 2D to 1D, make sure update
        # dp every time
        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    dp[j] = True
                elif i == 0:
                    dp[j] = (s2[:j] == s3[:j])
                elif j == 0:
                    dp[j] = (s1[:i] == s3[:i])
                else:
                    dp[j] = s1[i - 1] == s3[i + j - 1] and dp[j] or \
                   s2[j - 1] == s3[i + j - 1] and dp[j - 1]
                # print(dp)
        return dp[-1]
```

### [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

```python
class Solution:
    # Pattern: DP - String Matching
    #   use dp[i][j] to match each location i, j in s1 and s2
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N = len(text1), len(text2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                    
        return dp[-1][-1]
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N = len(text1), len(text2)
        # dp is for a i in text1, the answer for text1[i] and text2[:] 
        dp = [0] * (N + 1)
        
        for i in range(1, M + 1):
            dp2 = [0] * (N + 1)
            for j in range(1, N + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp2[j] = dp[j - 1] + 1
                else:
                    dp2[j] = max(dp2[j - 1], dp[j])
            dp = dp2
                
        return dp[-1]
```

### [1092. Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/)

```python
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        # Pattern: DP - Longest Common Subsequence
        M, N = len(str1), len(str2)
        dp = [[''] * (N + 1) for _ in range(M + 1)]
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + str1[i - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
        
        lcs = dp[-1][-1]
        
        # Trick: Construct Shortest Common Superseq from LCS
        ans, i, j = "", 0, 0
        for c in lcs:
            while str1[i] != c:
                ans += str1[i]
                i += 1
            while str2[j] != c:
                ans += str2[j]
                j += 1
            ans += c
            i, j = i + 1, j + 1
            
        return ans + str1[i:] + str2[j:]
```

### [72. Edit Distance](https://leetcode.com/problems/edit-distance/)

```python
class Solution:
    # https://medium.com/@ethannam/understanding-the-levenshtein-distance-equation-for-beginners-c4285a5604f0
    def minDistance(self, word1: str, word2: str) -> int:
        M, N = len(word1), len(word2)
        if M * N == 0: return M + N
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(M + 1):
            dp[i][0] = i
        for j in range(N + 1):
            dp[0][j] = j
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # min of below possibilities
                    # replace i: dp[i - 1][j - 1] + 1
                    # delete i: dp[i - 1][j] + 1
                    # insert i: dp[i][j - 1] + 1
                    # (somehow make first i, and j - 1 same, then
                    # add same w2[j] in w1[i] position)
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]
```

### [115. Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # Pattern: DP - String Matching
        # 1. Demention of dp is 1 + len()
        # 2. Loop for i, j start from 1 end len + 1
        # 3. s[i - 1] not s[i]
        M, N = len(s), len(t)
        if not M: return 0
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        # Initial value: for every source, if target is empty,
        # dp is 1
        for i in range(M + 1):
            dp[i][0] = 1
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # dp[i][j] must include max num of subseq in dp[i - 1][j]
                # same target, pre source withouth i-th
                dp[i][j] = dp[i - 1][j]
                
                # If the char matches, all max num of pre target in pre
                # source counts
                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        
        return dp[-1][-1]
```

### [712. Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/)

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        M, N = len(s1), len(s2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        # Initate all states where one of s is empty
        for i in range(1, M + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(1, N + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        
        # Start from first char of both
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # You know cost of make [i - 1] == [j], then
                    # cost of make [i] = [j] is just cost of[i - 1]
                    # plus cost ord([i])
                    dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]), 
                                   dp[i][j - 1] + ord(s2[j - 1]))
        return dp[-1][-1]
```

### [727. Minimum Window Subsequence](https://leetcode.com/problems/minimum-window-subsequence/)

```python
class Solution:
    def minWindow(self, S: str, T: str) -> str:
        # Pattern: DP - two arrays
        # Intuition: dp[i][j] is min len subseq in [0,i] of S contains
        # sebseq [0,j] of T. If we knon mix dp[i][N], we know idx i is
        # the last idx of min subseq in S that contains T, and we know
        # the min len, then we can know the first idx (pos - l)
        M, N = len(S), len(T)
        dp = [[math.inf] * (N + 1) for _ in range(M + 1)]
        # Trick: No need to do S[i - 1] anymore
        # S, T = '#' + S, '#' + T
        
        for i in range(M + 1):
            dp[i][0] = 0
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if S[i - 1] == T[j - 1]:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + 1
                else:
                    dp[i][j] = dp[i - 1][j] + 1
        
        l, pos = math.inf, 0
        for i in range(M + 1):
            if dp[i][N] < l:
                l = dp[i][N]
                pos = i

        return S[pos - l:pos] if l < math.inf else ""
```

### [583. Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        M, N = len(word1), len(word2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(M + 1):
            dp[i][0] = i
            for j in range(1, N + 1):
                if i == 0:
                    dp[0][j] = j
                else:
                    if word1[i - 1] == word2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                        
        return dp[-1][-1]
```

### [1035. Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/)

```python
class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        # Pattern: DP - two arrays - LCS
        M, N = len(A), len(B)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[-1][-1]
```

### [801. Minimum Swaps To Make Sequences Increasing](https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/)

```python
class Solution:
    def minSwap(self, A: List[int], B: List[int]) -> int:
        # https://youtu.be/__yxFFRQAl8
        N = len(A)
        # Intuition: For each pos i, swap/keep denotes min swaps needed
        # to keep A, B valid with/without i-th num swapped
        swap, keep = [math.inf] * N, [math.inf] * N
        swap[0], keep[0] = 1, 0
        
        for i in range(1, N):
            # All the [i - 1] state are valid!
            # Based on comparasion of A[i], A[i - 1], B[i], B[i - 1],
            # there are at most 2 possiblilityes for swap and keep,
            # try to compute all of them and get the min
            if A[i] > A[i - 1] and B[i] > B[i - 1]:
                # This is best guess at the moment, when we don't know
                # relation betwwen A[i] and B[i - 1]
                swap[i] = swap[i - 1] + 1
                keep[i] = keep[i - 1]
            if A[i] > B[i - 1] and B[i] > A[i - 1]:
                swap[i] = min(swap[i], keep[i - 1] + 1)
                keep[i] = min(keep[i], swap[i - 1])
                
        return min(swap[-1], keep[-1])
```


## Palindrome with insertion or deletion

### [1216. Valid Palindrome III](https://leetcode.com/problems/valid-palindrome-iii/)

```python
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        N = len(s)
        # dp[i][j] is min delete to makde [i,j] palindrome
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N - 2, -1, -1):
            for j in range(i + 1, N):
                if s[i] == s[j]:
                    # For each new i and it's first j, they are
                    # adjecent, so dp[i + 1][j - 1] is dp[j][i] (j >= i)
                    # in dp, dp[i][j] == 0 when i > j, so for i and
                    # first j (i + 1), the value is always 0, when
                    # s[i] == s[j]
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    # If s[i] != s[j], to make [i,j] palindrome, we
                    # needs to delete i or j, note not both i and j,
                    # cuz it is duplicate. You delete one, the other
                    # is handled in subproblem.
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1
        return dp[0][N - 1] <= k
```

### [1312. Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        N = len(s)
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N - 2, -1, -1):
            for j in range(i + 1, N):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1
        
        return dp[0][N - 1]
```

### [10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s, p = ' ' + s, ' ' + p
        LS, LP = len(s), len(p)
        dp = [[False] * LP for _ in range(LS)]
        
        # Because * cannot be first letter, so start from 2
        dp[0][0] = True
        for j in range(2, LP):
            if p[j] == '*':
                dp[0][j] = dp[0][j - 2]
        
        for i in range(1, LS):
            for j in range(1, LP):
                if p[j] in (s[i], '.'):
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j] == '*':
                    '''
                    If p[j] is '*', it can elimiate last char in s,
                    so we just need compare dp[i - 1][j].
                    
                    Case1:
                    abc
                    abcd*
                    
                    Case2:
                    abcd
                    abcd*
                    
                    abcdd
                    abcd*
                    
                    abcd
                    abc.*
                    
                    why not dp[i - 1][j - 2]?
                    Because we handle it in another case (else)
                    
                    '''
                    dp[i][j] = dp[i][j - 2]
                    if not dp[i][j] and p[j - 1] in (s[i], '.'):
                        dp[i][j] = dp[i - 1][j]
 
        return dp[-1][-1]
```

### [718. Maximum Length of Repeated Subarray](https://leetcode.com/problems/maximum-length-of-repeated-subarray/)

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        
        # Trick: [a] * n will replicate a's pointer n times, not copy of a n times!
        # dp = [[0] * (len(B) + 1)] * (len(A) + 1)
        # print(dp)
        
        dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        # print(dp)
        
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                if A[i] == B[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                    
        return max(max(row) for row in dp)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        M, N = len(nums1), len(nums2)
        # dp[i][j] is max len of substr ending with i and j
        # for nums1[0:i+1] and nums2[0:j+1]
        # last
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        ans = 0
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                ii, jj = i - 1, j - 1
                if nums1[ii] == nums2[jj]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = 0
                ans = max(ans, dp[i][j])
        return ans
```