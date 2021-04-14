
### [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # Pattern: DP - On a single string
        N, ans = len(s), 0
        dp = [[False] * N for _ in range(N)]
        
        for i in range(N):
            dp[i][i] = True
            ans += 1
            if i < N - 1:
                dp[i][i + 1] = (s[i] == s[i + 1])
                ans += dp[i][i + 1]
        
        # Use len as the support point
        for l in range(3, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # Trick: Check parlindrom of shorter substr
                dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
                ans += dp[i][j]
        
        return ans
```

### [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # Pattern: Another way other than using l 
        # https://leetcode.com/problems/longest-palindromic-subsequence/discuss/99101/Straight-forward-Java-DP-solution
        N = len(s)
        dp = [[1] * N for _ in range(N)]
        
        for i in range(N - 1):
            dp[i][i + 1] = 1 + (s[i] == s[i + 1])
            
        for l in range(3, N + 1):
            for i in range(0, N - l + 1):
                j = i + l - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][-1]
```



### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```python
class Solution:
    
    def longestPalindrome(self, s: str) -> str:
        def expand(i):
            centers = [(i, i), (i, i + 1)]
            max_l = (0, -1, -1)
            
            for c in centers:
                j, k = c[0], c[1]
                while j >=0 and k < len(s) and s[j] == s[k]:
                    j -= 1
                    k += 1
                # -1 because j and k is 2 unit longer than the valid len
                if k - j - 1 > max_l[0]:
                    max_l = (k - j -1, j + 1, k -1)
            return max_l
                
            
        max_l = (0, -1, -1)
        for i in range(len(s)):
            l, s_, e = expand(i)
            if l > max_l[0]:
                max_l = (l, s_, e)
        
        print(max_l)
        return s[max_l[1]:max_l[2] + 1]
    
    def longestPalindrome(self, s: str) -> str:
        N = len(s)
        if N in (0, 1): return s
        dp = [[0] * N for _ in range(N)]
        # If no further updates, this is the answer
        ans, ans_len = s[0], 1
        
        for i in range(N):
            dp[i][i] = True
            if i < N - 1:
                dp[i][i + 1] = (s[i] == s[i + 1])
                if dp[i][i + 1] and ans_len == 1:
                    ans = s[i:i + 2]
                    ans_len = 2
        
        for l in range(3, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                dp[i][j] = dp[i + 1][j - 1] and s[i] == s[j]
                if dp[i][j] and l > ans_len:
                    ans = s[i:j + 1]
                    ans_len = l
        
        return ans
```


### [1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)

```python
class Solution:
    def mctFromLeafValues(self, A):
        # https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/discuss/339959/One-Pass-O(N)-Time-and-Space
        ans = 0
        while len(A) > 1:
            i = A.index(min(A))
            # Trick: why use substring sum? - avoid edge cases
            ans += min(A[i - 1:i] + A[i + 1:i + 2]) * A.pop(i)
        return ans
    
    def mctFromLeafValues(self, A):
        N = len(A)
        # dp[i][j] is the answer for substr [i][j]
        dp = [[math.inf] * N for _ in range(N)]
        
        # Initial value: single value substr yield 0
        for i in range(N):
            dp[i][i] = 0
        
        # Trick: Cannot simple use 2 loop for i and j, in that way not work
        #   First process all the sub str starting from 2, when len is 2
        #   and partition by k, the substr len is 1 which has value 0, then
        #   all len 3 has partion of len 1 or 2 which also calculated.
        for l in range(2, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # 0 or 2 children
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], 
                                  max(A[i:k + 1]) * max(A[k + 1:j + 1]) + dp[i][k] + dp[k + 1][j])

        return dp[0][N - 1]
    
    def mctFromLeafValues(self, A):
        @functools.lru_cache(None)
        def dp(i, j):
            if i == j: return 0
            ans = math.inf
            for k in range(i, j):
                ans = min(ans, max(A[i:k+1]) * max(A[k+1:j+1]) + dp(i, k) + dp(k + 1, j))
            return ans
        return dp(0, len(A) - 1)
```

### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
                
        return dp[-1]
```


### [1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/)

```python
#
#    |  0  1  2  3
#  --------------------
#  0 |  0  0  84 
#  1 |     0  0  140
#  2 |        0  0 
#  3 |  0        0
# 
#

class Solution(object):
    def minScoreTriangulation(self, A):
        # Pattern: Partition
        # For each subarray i~j, pick k in btw, i and j are disconnected
        N = len(A)
        dp = [[0] * N for _ in range(N)]
        
        # The length is num of edge not 
        # num of nodes! So it is 2 ~ (N - 1),
        # this makes the -/+ l clean
        for l in range(2, N):
            for i in range(N - l):
                j = i + l
                # For each k btw i and j exclusive
                dp[i][j] = math.inf
                for k in range(i + 1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i] * A[k] * A[j])
        
        return dp[0][-1]
```


### [546. Remove Boxes](https://leetcode.com/problems/remove-boxes/)

```python
class Solution:
    # https://leetcode.com/problems/remove-boxes/discuss/101310/Java-top-down-and-bottom-up-DP-solutions
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        dp = [[[0] * N for _ in range(N)] for _ in range(N)]
        # The initial states for k are based on assumption,
        # during iteration to populate true value, only the real
        # case will be visited
        for i in range(N):
            # boxes[i] is the (i+1)-th box, so there is i
            # possible boxes to the left of boxes[i], so the loop
            # should include i for k
            for k in range(i + 1):
                dp[i][i][k] = (k + 1) * (k + 1)
        
        # Trick: If l starts from 1 to N - 1, the code is cleaner
        # Trick: Interate from smaller len, so DP can work
        for l in range(1, N):
            for j in range(l, N):
                i = j - l
                for k in range(i + 1):
                    # If remove boxes[i]
                    dp[i][j][k] = (k + 1) * (k + 1) + dp[i + 1][j][0]
                    # If keep boxes[i]
                    # Try all values for i < m <= j where i and m same color
                    # dp[i][j] = (max value if delete all btw i and m exclusive) + 
                    # (max value of delete boxes[i] + boxes[m, j+1])
                    for m in range(i + 1, j + 1):
                        if boxes[m] == boxes[i]:
                            dp[i][j][k] = max(dp[i][j][k], dp[i + 1][m - 1][0] + dp[m][j][k + 1])

        return dp[0][N -1][0]
    
    def removeBoxes(self, boxes: List[int]) -> int:
        @functools.lru_cache(None)
        def dp(i, j, k):
            if i > j:
                return 0

            # prune the calculation
            while i + 1 <= j and boxes[i + 1] == boxes[i]:
                i += 1
                k += 1

            res = (k + 1) * (k + 1) + dp(i + 1, j, 0)
            for t in range(i + 1, j + 1):
                if boxes[t] == boxes[i]:
                    res = max(res, dp(i + 1, t - 1, 0) + dp(t, j, k + 1))
            return res

        return dp(0, len(boxes) - 1, 0)
```

### [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        
        # dp[i][j] is the max coins if adding baloons back to 
        # between i and j, i and j are not included. The location of baloons
        # added to btw i,j is decided by nums. Spaces btw i,j is currently
        # empty.
        dp = [[0] * n for _ in range(n)]
        
        # The current n is after reframe, so n - 1 is the dummy right node,
        # we want to calculate left: 0 ~ n-2, because left has to be less than right
        # right should be left + 2 ~ n - 1, because the purpose of left/right is to 
        # define walls for real baloon location to add baloons.
        # When left is n - 2, no right avaialbe, and no realy baloon locations, so
        # the 2nd loop won't run. 
        # To calculate max coin btw left/right exclusive, interate lef+1~right-1,
        # and calculate if put baloon there, and plus max coin of fill the two sub
        # section.
        for left in range(n - 2, -1, -1):
            for right in range(left + 2, n):
                dp[left][right] = max(nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right] for i in range(left+1, right))
                
        return dp[0][n - 1]
```

### [1000. Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/)

```python
class Solution:
    def mergeStones(self, stones: List[int], K: int) -> int:
        N = len(stones)
        if (N - 1) % (K - 1): return -1
        dp = [[0] * N for _ in range(N)]
        
        for l in range(K, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                dp[i][j] = math.inf
                for k in range(i, j, K - 1):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j])
                if (l - 1) % (K - 1) == 0:
                    dp[i][j] += sum(stones[i:j + 1])
        
        return dp[0][-1]
    
    def mergeStones(self, stones: List[int], K: int) -> int:
        N = len(stones)
        if (N - 1) % (K - 1): return -1
        prefix = [0] * (N+1)
        for i in range(1,N+1): prefix[i] = stones[i-1] + prefix[i-1]
        dp = [[0] * N for _ in range(N)]
        for m in range(K, N+1):
            for i in range(N-m+1):
                dp[i][i+m-1] = min(dp[i][k] + dp[k+1][i+m-1] for k in range(i, i+m-1, K-1)) + (prefix[i+m] - prefix[i] if (m-1)%(K-1) == 0 else 0)
        return dp[0][N-1]
    
    def mergeStones(self, stones: List[int], K: int) -> int:
        N = len(stones)
        # Inuition: dp[i][j][k] denotes min cost to partition [i,j] into k
        # groups where only merge of 1 or K piles are allowed
        dp = [[[math.inf] * (K + 1) for _ in range(N)] for _ in range(N)]
        
        # DP partition 2 - iterate from small segments to bigger ones
        for l in range(1, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # For each [i, j], calculate min cost for [1. K] partitions,
                # when l == 1, dp[1] = 0
                # when k == 1, dp[k] = dp[K] + sum[i:j+1];
                # when k >= 2, dp[k] = ...
                # For each [i, j] followig the merge rule, it is possible for [1, k] to
                # gave valid cost, but not always depends on nums
                if l == 1:
                    dp[i][j][1] = 0
                else:
                    for k in range(2, min(K, l) + 1):
                        for m in range(i + k - 1, j + 1):
                            dp[i][j][k] = min(dp[i][j][k], dp[i][m - 1][k - 1] + dp[m][j][1])
                    dp[i][j][1] = dp[i][j][K] + sum(stones[i:j + 1])

        return dp[0][N - 1][1] if dp[0][N - 1][1] < math.inf else -1
```

### [375. Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/)

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # Pattern: DP - partition
        # Intuition: Given a number nn, we have to find the worst case cost of 
        #   guessing a number chosen from the range (1, n)(1,n), assuming that 
        #   the guesses are made intelligently(minimize the total cost).
        #   dp[i][j] is the answer for substr [i, j] both inclusive,
        #   the maxmin cost is the cost assume you find the correct num at
        #   the very last. So if their is only one num, cost is 0, cuz it has
        #   to be left, if more than 1, first pick one (wrong) number other than
        #   last one, split to 3 parts. For len = 2, pick the first smaller num.
        #   
        #   The size of dp is (n + 1), because for dp[i][k - 1] when i = 1, if i start
        #   from 0, it become dp[0][-1], not correct
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for l in range(2, n + 1):
            for i in range(1, n - l + 1 + 1):
                j = i + l - 1
                dp[i][j] = math.inf
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], k + max(dp[i][k - 1], dp[k + 1][j]))
        
        return dp[1][n]
```

### [1246. Palindrome Removal](https://leetcode.com/problems/palindrome-removal/)

```python
class Solution:
    def minimumMoves(self, arr: List[int]) -> int:
        N = len(arr)
        dp = [[0] * (N + 1) for _ in range(N + 1)]
        
        for l in range(1, N + 1):
            for i in range(N - l + 1):
                j = i + l - 1
                # Trick: Use l == 1 to set initial value
                if l == 1:
                    dp[i][j] = 1
                else:
                    # Case 1: i is independently removed, not same to any else
                    dp[i][j] = dp[i + 1][j] + 1
                    
                    # Case 2: i and i + 1 same
                    if arr[i] == arr[i + 1]:
                        dp[i][j] = min(dp[i][j], dp[i + 2][j] + 1)
                    
                    # Case 3: i and k same where k in [i + 2, j]
                    for k in range(i + 2, j + 1):
                        if arr[i] == arr[k]:
                            # Why no + 1 when partioned by k? Bcuz when arr[i]==arr[k],
                            # i and k can be removed together with last remove of 
                            # [i + 1, k - 1], because last remove is a palindrome!
                            dp[i][j] = min(dp[i][j], dp[i + 1][k - 1] + dp[k + 1][j])
        
        return dp[0][N - 1]
```

### [1770. Maximum Score from Performing Multiplication Operations](https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/)

```python
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        m, n = len(multipliers), len(nums)
        dp = [[-math.inf] * (m + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        
        ans = -math.inf
        for l in range(1, m + 1):
            for i in range(0, m + 1):
                j = l - i
                if i >= 1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j] + nums[i - 1] * multipliers[i + j - 1])
                if j >= 1:
                    dp[i][j] = max(dp[i][j], dp[i][j - 1] + nums[n - j] * multipliers[i + j - 1])
        
                if l == m:
                    ans = max(ans, dp[i][j])
                    
        return ans
    
    def maximumScore(self, nums, muls):
        # let dp[i][j] represents pick i elements from left and pick j from right from nums
        # dp[i][j] = Max(dp[i-1][j] + muls[i+j-1] * nums[i-1], dp[i][j-1] + muls[i+j-1] * nums[n-j])
        n, m = len(nums), len(muls)
        dp = [[0] * (m+1) for _ in range(m+1)]
        res = float("-inf")
        for i in range(0, m+1):
            for j in range(0, m-i+1):
                if i == 0 and j == 0: 
                    continue
                l, r = float("-inf"), float("-inf")
                if i > 0: l = dp[i-1][j] + muls[i+j-1] * nums[i-1] # pick left
                if j > 0: r = dp[i][j-1] + muls[i+j-1] * nums[n-j] # pick right
                dp[i][j] = max(l, r)
                if i + j == m:
                    res = max(res, dp[i][j])
        return res
```