

## Intervals

### [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # Sort it so don't worry starting index
        intervals.sort(key=lambda x: x[0])
        
        # Keep the running result
        merged = []
        
        for it in intervals:
            
            # Only compare endding index
            if merged and merged[-1][1] >= it[0]:
                # Need max because [1,4], [2,3]
                merged[-1][1] = max(it[1], merged[-1][1])
            else:
                merged += [it]
                
        return merged
```

### [57. Insert Interval](https://leetcode.com/problems/insert-interval/)

```python
class Solution(object):
    def insert(self, intervals, newInterval):
        res, n = [], newInterval
        for index, i in enumerate(intervals):
            if i[-1] < n[0]:
                res.append(i)
            elif n[-1] < i[0]:
                res.append(n)
                return res+intervals[index:]  # can return earlier
            else:  # overlap case
                n[0] = min(n[0], i[0])
                n[-1]= max(n[-1], i[-1])
        res.append(n)
        return res
```

### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        rooms = []
        
        intervals.sort(key=lambda x: x[0])
        
        for it in intervals:
            if rooms and it[0] >= rooms[0]:
                    heapq.heappop(rooms)
                
            heapq.heappush(rooms, it[1])
                
        return len(rooms)
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

### [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        
        mp = {}
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            
            if k != 0:
                sum = sum % k
            
            if not sum and i > 0:
                return True
            
            if sum in mp:
                if i - mp[sum] > 1:
                    return True
            else:
                mp[sum] = i
            
        return False
```

## Index as Hash Key

### [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # Intuition: There are N pos in list, if put 1..N, then
        # missing is N + 1, if there is any num other than 1..N,
        # (<=0 or >N), num in 1..N will be kicked out and the 
        # anwser must be in 1..N. Use index of array to refresent
        # num 1..N, if i exist, nums[i] = -nums[i], it will keep
        # value of nums[i] and use minus to makr existence of i.
        
        # Trick: Roll out 1 first, cuz 1 will be used for nums which is 
        # negative or > N
        if 1 not in nums:
            return 1
        
        N = len(nums)
        
        # Normalize other numbers into 1, all values must be 1..N
        # to reflect the scope of the index
        for i in range(N):
            if nums[i] <= 0 or nums[i] > N:
                nums[i] = 1
        
        # Trick: Index as hash key
        # Mark the existence of 1..N, always use abs(), cuz the
        # value could be mark as negative by previous steps. Use
        # index 0 for N, cuz index is 0-based.
        for i in range(N):
            x = abs(nums[i])
            if x == N:
                nums[0] = -abs(nums[0])
            else:
                nums[x] = -abs(nums[x])
        
        # Check 2..N - 1 first
        for i in range(2, N):
            if nums[i] > 0:
                return i
        
        # Check N
        if nums[0] > 0:
            return N
        
        # If 1..N all exist, return N + 1
        return N + 1
```

### [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

```python
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for x in nums:
            if nums[abs(x)-1] < 0:
                res.append(abs(x))
            else:
                nums[abs(x)-1] *= -1
        return res
```

## Circular Array

### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

### [539. Minimum Time Difference](https://leetcode.com/problems/minimum-time-difference/)

```python
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        def convert(time):
            return int(time[:2]) * 60 + int(time[3:])
        
        times = sorted(map(convert, timePoints))
        
        # Trick: Differences of recursive array
        # Trick: % (a day's minutes) can fix the 00:00 and 24:60
        return min((y - x) % (24 * 60)  for x, y in zip(times, times[1:] + times[:1]))
```

## Others

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

### [245. Shortest Word Distance III](https://leetcode.com/problems/shortest-word-distance-iii/)

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
### [845. Longest Mountain in Array](https://leetcode.com/problems/longest-mountain-in-array/)
### [941. Valid Mountain Array](https://leetcode.com/problems/valid-mountain-array/)

```python
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        N, i = len(arr), 0
        
        while i < N - 1 and arr[i] < arr[i + 1]:
            i += 1
        
        if i == 0 or i == N - 1:
            return False
        
        while i < N - 1 and arr[i] > arr[i + 1]:
            i += 1
            
        return i == N - 1
```

### [6. ZigZag Conversion](https://leetcode.com/problems/zigzag-conversion/)

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        
        if numRows == 1:
            return s
        
        rows = [''] * numRows
        
        cur_row, next_step = 0, 1
        
        for c in s:
            rows[cur_row] += c
            
            if (cur_row == 0 and next_step == -1) or (cur_row == numRows - 1 and next_step == 1):
                next_step = -next_step
            
            cur_row += next_step
            
        return ''.join(rows)
```

### [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        # two get pointers for nums1 and nums2
        p1 = m - 1
        p2 = n - 1
        # set pointer for nums1
        p = m + n - 1
        
        # while there are still elements to compare
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] =  nums1[p1]
                p1 -= 1
            p -= 1
        
        # add missing elements from nums2
        nums1[:p2 + 1] = nums2[:p2 + 1]
```

### [767. Reorganize String](https://leetcode.com/problems/reorganize-string/)

```python
class Solution(object):
    def reorganizeString(self, S):
        N = len(S)
        A = []
        for c, x in sorted((S.count(x), x) for x in set(S)):
            if c > (N+1)/2: return ""
            A.extend(c * x)
        ans = [None] * N
        ans[::2], ans[1::2] = A[N//2:], A[:N//2]
        return "".join(ans)
```

### [1054. Distant Barcodes](https://leetcode.com/problems/distant-barcodes/)

```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        count = collections.Counter(barcodes)
        barcodes.sort(key=lambda c: (count[c], c))
        barcodes[::2], barcodes[1::2] = barcodes[len(barcodes)//2:], barcodes[:len(barcodes)//2]
        return barcodes
```

### [1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

```python
class Solution:
    def maxSumTwoNoOverlap(self, A, L, M):
        for i in range(1, len(A)):
            A[i] += A[i - 1]
        res, Lmax, Mmax = A[L + M - 1], A[L - 1], A[M - 1]
        for i in range(L + M, len(A)):
            Lmax = max(Lmax, A[i - M] - A[i - L - M])
            Mmax = max(Mmax, A[i - L] - A[i - L - M])
            res = max(res, Lmax + A[i] - A[i - M], Mmax + A[i] - A[i - L])
        return res
```

### [1146. Snapshot Array](https://leetcode.com/problems/snapshot-array/)

```python
class SnapshotArray(object):

    def __init__(self, n):
        self.A = [[[-1, 0]] for _ in xrange(n)]
        self.snap_id = 0

    def set(self, index, val):
        self.A[index].append([self.snap_id, val])

    def snap(self):
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index, snap_id):
        i = bisect.bisect(self.A[index], [snap_id + 1]) - 1
        return self.A[index][i][1]
        


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)
```

### [1570. Dot Product of Two Sparse Vectors](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/)

```python
class SparseVector:
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec):
        """
        :type vec: 'SparseVector'
        :rtype: int
        """
        ans = 0
        for i in range(len(self.nums)):
            ans += self.nums[i] * vec.nums[i]
        return ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
```



