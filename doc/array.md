### [244. Shortest Word Distance II](https://leetcode.com/problems/shortest-word-distance-iii/)

```python
class Solution:
    def shortestWordDistance(self, words: List[str], word1: str, word2: str) -> int:
        idx = -1
        ans = math.inf
        for i in range(len(words)):
            if words[i] == word1 or words[i] == word2:
                if idx != -1 and (word1 == word2 or words[i] != words[idx]):
                    ans = min(ans, i - idx)
                idx = i
                
        return ans
```

### [244. Shortest Word Distance II](https://leetcode.com/problems/shortest-word-distance-ii/)

```python
class WordDistance:

    def __init__(self, words: List[str]):
        self.locations = defaultdict(list)
        for i, w in enumerate(words):
            self.locations[w].append(i)

    def shortest(self, word1: str, word2: str) -> int:
        # Trick: Two Pointers
        #        The answer can be enumerated by two pointers,
        #        and every step, moving one pointer can leads to
        #        a optimal answer.
        loc1, loc2 = self.locations[word1], self.locations[word2]
        i, j = 0, 0
        min_dis = math.inf
        while i < len(loc1) and j < len(loc2):
            min_dis = min(min_dis, abs(loc1[i] - loc2[j]))
            if loc1[i] < loc2[j]:
                i += 1
            else:
                j += 1
        return min_dis
```

### [311. Sparse Matrix Multiplication](311. Sparse Matrix Multiplication)

```python
class Solution:
    def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        Ra, Ca, Rb, Cb = len(A), len(A[0]), len(B), len(B[0])
        
        ans = [[0] * Cb for _ in range(Ra)]
        
        for r in range(Ra):
            for c in range(Cb):
                for i in range(Ca):
                    ans[r][c] += A[r][i] * B[i][c]
        
        return ans
```

### [777. Swap Adjacent in LR String](https://leetcode.com/problems/swap-adjacent-in-lr-string/)

```python
class Solution(object):
    def canTransform(self, start, end):
        # It is necessary condition that 2 str are same without X,
        # so it the condition is not true, return false
        if start.replace('X',  '') != end.replace('X', ''):
            return False
        
        # Use i and j to traverse and compare start and end,
        # fastfoward i and j if they points to 'X'.
        # The should always pointing to same value, as L and R cannot
        # be swapped. And the current L in start has to be after cur L in
        # end, because it can only go left. And the cur R in start has to 
        # be before cur R in end, because it can only go right.
        N = len(start)
        i = j = 0
        while i < N and j < N:
            # Trick: Skip values
            while i < N and start[i] == 'X':
                i += 1
            while j < N and end[j] == 'X':
                j += 1
                
            if i == N or j == N:
                return i == j
            
            if start[i] != end[j]:
                return False
            
            if start[i] == 'L' and i < j:
                return False
            
            if start[i] == 'R' and i > j:
                return False
            
            i += 1
            j += 1
        
        return True 
```



## Subarray sum

### [363. Max Sum of Rectangle No Larger Than K](https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/)
```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        R, C = len(matrix), len(matrix[0])
        
        # Trick: Get max sum of subarrays no larger than k
        def max_sum_no_larger_than_k(arr, k):
            prefix_sums = [math.inf]
            cur_sum = 0
            ans = -math.inf
            for x in arr:
                bisect.insort(prefix_sums, cur_sum)
                cur_sum += x
                i = bisect.bisect_left(prefix_sums, cur_sum - k)
                ans = max(ans, cur_sum - prefix_sums[i])
            return ans
        
        # Trick: Rotate the matrix if R > C
        # Cuz the time complexity is O(R*R*C*logC), so if C is longer, the total
        # complexity will be reduced greatly. So if R is longer, rotate it.
        if R > C:
            matrix2= [[None] * R for _ in range(C)]
            for r in range(R):
                for c in range(C):
                    matrix2[c][r] = matrix[r][c]
            matrix, R, C = matrix2, C, R
        
        # Trick: Use prefix-sums to get sum of each sub-matrix
        # Update matrix in place, calculate prefix sum for each column
        for r in range(R - 1):
            for c in range(C):
                matrix[r + 1][c] += matrix[r][c]
        
        ans = -math.inf
        for r1 in range(R):
            for r2 in range(r1, R):
                arr = [matrix[r2][c] - (matrix[r1][c] if r2 > r1 else 0) for c in range(C)]
                ans = max(ans, max_sum_no_larger_than_k(arr, k))
        
        return ans
```

### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cur_sum, ans = 0, 0
        mp = defaultdict(int)
        # This initial value is critical, because
        # the way is to find the element AFTER which
        # all items until cur item sum to k. But what
        # if items starting from first to cur? So we
        # need key as 0 so when (cur_sum - k) there is
        # a match, which means sum of 0..cur sums to k
        mp[0] = 1
        for x in nums:
            cur_sum += x
            ans += mp[cur_sum - k]
            mp[cur_sum] += 1
        return ans
```

### [1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        
        def num_of_sum(arr, k):
            cur_sum, ans = 0, 0
            mp = defaultdict(int)
            mp[0] = 1
            for x in arr:
                cur_sum += x
                ans += mp[cur_sum - k]
                mp[cur_sum] += 1
            return ans
        
        R, C = len(matrix), len(matrix[0])
        
        for r in range(1, R):
            for c in range(C):
                matrix[r][c] += matrix[r - 1][c]
        
        ans = 0
        for r1 in range(R):
            for r2 in range(r1, R):
                arr = [matrix[r2][c] - (matrix[r1][c] if r2 != r1 else 0) for c in range(C)]
                ans += num_of_sum(arr, target)
        return ans
```

### [325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        # Pattern: Subarray sum of K
        # In this case the answer is able max length, the presence
        # of the sum key matters, so unlike the max nums we just need 
        # to count the num where 0 is ok
        cur_sum, ans = 0, 0
        mp = {0: -1}
        for i, x in enumerate(nums):
            cur_sum += x
            
            if cur_sum - k in mp:
                ans = max(ans, i - mp[cur_sum - k])
            
            mp[cur_sum] = mp.get(cur_sum, i)
        
        return ans
```

### [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)

```python
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        # https://leetcode.com/problems/subarray-sums-divisible-by-k/discuss/310767/(Python)-Concise-Explanation-and-Proof
        
        # accounts[0] == 1 for edge case runnsing_sum%k == 0, 
        # you dont need 2 subarray with save value, and every new 0 value will 
        # add prev_account+1, 1 is the new prefix sum itself
        accounts = [1] + [0] * (K - 1)
        
        running_sum = 0
        ans = 0
        for n in A:
            running_sum += n
            
            key = running_sum % K
            
            # A new key value will result in accounts[key] MORE matches to be added
            # to the results (accounts[key] is the previous value for key)
            ans += accounts[key]
            
            accounts[key] += 1
            
        return ans
    
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        running_sum, ans = 0, 0
        mp = defaultdict(int)
        mp[0] = 1
        for x in A:
            running_sum += x
            key = running_sum % K
            ans += mp[key]
            mp[key] += 1
        return ans
```