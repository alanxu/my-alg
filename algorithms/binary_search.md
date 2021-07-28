#
It can apply to an item in a seq where the left of the item meet a condition while the right of item meet the composite
condition;
Finds smallest item matching a condition -> Find the lower_bound of a scope mating a condition -> Use alg works for
scope [i, j] or [i, j), which is close-close or close-open
Close-open, close-close both can work for the problems, the final result of C-O is left and right equal, while C-C is 
left or right (the final state is right at left, left at right)

Close-Open:
1. while left < right
2. mid = left + (right - left) // 2, this is to avoid overflow for Java/C++, Python is fine
3. if condition: right = mid
4. else: left = mid + 1
5. the final location = left = right

Close-Close
1. while left <= right
2. mid = left + (right - left) // 2, this is to avoid overflow for Java/C++, Python is fine
3. if condition: right = mid - 1
4. else: left = mid + 1
5. the final location = left or right, the final state is (right, left)


## Template 1

### [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2: return x
        hi, lo = x // 2, 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if mid ** 2 == x:
                return mid
            if mid ** 2 < x:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return hi
```

### [374. Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/)

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        lo, hi = 1, n
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            guess_ = guess(mid)
            if  guess_ == 0:
                return mid
            if guess_ == -1:
                hi = mid - 1
            else:
                lo = mid + 1
        return -1
```

### [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        def get_partition(x):
            ''' Which partition the num is at. '''
            return 1 if x > nums[-1] else 2
        
        left, right = 0, len(nums) - 1
        target_par = get_partition(target)

        while left <= right:
            mid = left + (right - left) // 2
            mid_par = get_partition(nums[mid])
            if nums[mid] == target:
                return mid
            # It is about determin mid and target which one is
            # ahead. So if mid and target are in same partition,
            # smaller one is ahead, if in diff partition, bigger
            # one is ahead. Then we know how to move the pointers.
            if target_par == mid_par and nums[mid] < target or \
                target_par != mid_par and nums[mid] > target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```
### [702. Search in a Sorted Array of Unknown Size](https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/)

```python
class Solution:
    def search(self, reader, target):
        left, right = 0, 2147483647
        
        while left <= right:
            mid = left + (right - left) // 2
            x = reader.get(mid)
            if x == target:
                return mid
            elif x < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left if reader.get(left) == target else -1
        
    def search(self, reader, target):
        if reader.get(0) == target:
            return 0
        
        # Trick: Search boundary for target
        left, right = 0, 1
        while reader.get(right) < target:
            left = right
            right <<= 1
        
        while left <= right:
            mid = left + (right - left) // 2
            x = reader.get(mid)
            if x == target:
                return mid
            elif x < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
```

### [367. Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/)

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        left, right = 2, num // 2
        while left <= right:
            x = left + (right - left) // 2
            sqr = x * x
            if sqr == num:
                return True
            elif sqr < num:
                left = x + 1
            else:
                right = x - 1
        
        return False
```

## Template 2

### [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)

```python
class Solution:
    def firstBadVersion(self, n):
        if n <= 1:
            return n
        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

### [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/)

```java
public class Solution {
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > nums[mid + 1])
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }
}
```

### [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

```python
# Fav
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # Binary Search template 2
        # The key of binary search is to find two opposite conditions
        # that we can move left and right, and the target is at the
        # the border of the two conditions
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            # When the target condition match, set right at mid, otherwise
            # move left to skip mid
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]
```

### [154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

```python
# Fav
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            # if nums[mid] == nums[right], there are 2 possibilities
            # - if nums[left] == nums[right], we don't know where is 
            #   the mid, consider cases [3, 3, 1, 3] and [3, 1, 3, 3],
            #   in this case, we move both left, right by 1, for edge case
            #   left and right are adjecent, left/right will swap and 
            #   loop will terminate, it doesn't impact the result cuz
            #   nums[left] == nums[right]
            # - if nums[left] > nums[right], right needs to move, bcuz
            #   if mid is on left side nums[mid] >= nums[left] > nums[right] 
            #   which conflicts with the initial condition of 
            # - it is impossible nums[left] < nums[right], cuz pointers will
            #   meet at the point of border
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            # if mid failed at right side
            elif nums[mid] <= nums[right]:
                right = mid
            # otherwise mid must on left side
            else:
                left = mid + 1
        return nums[left]
```


### [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binary_search(search_leftmost):
            # hi set to len(nums), not len(nums) - 1, for input [], it can return correct value
            # it wont impact anything else
            lo, hi = 0, len(nums)
            
            # Trick: Binary Search
            # lo always (mid + 1)
            # mid always fails on lo if they are adjenct 
            # when looking for left, hi should move when equal
            # when looking for right, lo should move when equal
            # when looking for right, because lo move when equal and lo = mid + 1, 
            # so lo/hi will meet at index nums[index] > target
            
            # Trick: no equal, because last operation will make pointer finally equal
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if search_leftmost:
                    if nums[mid] >= target:
                        hi = mid
                    else:
                        lo = mid + 1
                elif not search_leftmost:
                    if nums[mid] <= target:
                        # Trick: low's next should alwasys different (mid + 1), otherwise
                        # you will be in deadloop. Because lo + (hi - lo) // 2 tends to use
                        # low value
                        lo = mid + 1
                    else:
                        hi = mid
            
            return lo
        
        left = binary_search(True)
        
        if left == len(nums) or nums[left] != target:
            return [-1, -1]
        
        return [left, binary_search(False) - 1]
```

### [658. Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/)

```python
class Solution:
    def findClosestElements1(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - 1
        while right - left + 1 > k:
            if abs(arr[left] - x) > abs(arr[right] - x):
                left += 1
            else:
                right -= 1
        return arr[left:right+1]
    
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        arr.sort(key=lambda n: abs(n - x))
        return sorted(arr[:k])
```

### [744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # Alg: Binary Search template 2
        left, right = 0, len(letters) - 1
        while left < right:
            mid = left + (right - left) // 2
            if letters[mid] > target:
                right = mid
            else:
                left = mid + 1
        return letters[left] if letters[left] > target else letters[0]
    
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        pos = bisect.bisect_right(letters, target)
        return letters[pos % len(letters)]
    
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # Alg: Binary Search template 1
        left, right = 0, len(letters) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if letters[mid] > target:
                right = mid - 1
            else:
                left = mid + 1

        pos = left if left < len(letters) else right
        return letters[pos] if letters[pos] > target else letters[0]
```



### [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)
```python
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def possible(guess):
            # Is it poosible to have k - 1 distances smaller than
            # guess?
            # Why left is global? Cuz the goal is to find num
            # of diff smaller than guess, if left is larger for
            # cur right, it is larger for next right, so no need
            # to start from 0
            count = left = 0
            for right in range(len(nums)):
                while nums[right] - nums[left] > guess:
                    left += 1
                count += right - left
            return count >= k
        
        nums.sort()
        lo, hi = 0, nums[-1] - nums[0]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if possible(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo
```

### [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

```python
class Solution:
    def splitArray(self, nums: List[int], K: int) -> int:
        # TLE
        N = len(nums)
        dp = [[math.inf] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        
        for i in range(1, N + 1):
            for k in range(1, min(K, i) + 1):
                dp[i][k] = math.inf
                prefix_sum = 0
                for j in range(i, k - 1, -1):
                    # When k is 1, j has to be 1, but we still do iteration,
                    # because when k == 0, dp[i][k] = inf except when i = 0,
                    # so the interation is not inf only when j = 1 (first element)
                    prefix_sum += nums[j - 1]
                    dp[i][k] = min(dp[i][k], max(dp[j - 1][k - 1], prefix_sum))
        
        return dp[-1][-1]
        
    def splitArray(self, nums: List[int], K: int) -> int:
        # Alg: Binary Search
        # Intuition: Search for min max sum of one partition, left is max of nums, right
        # is sum of nums. The lager of the ans, the fewer the partions. The answer is
        # the min value can keep <= K partitions, decrease by 1, the partitions will > K.
        # So it is the edge value that can be found by binary search.
        # The edge case is gauanteed to make l and r meet.
        N = len(nums)
        l = r = 0
        
        for x in nums:
            r += x
            l = x if x > l else l
        
        ans = r
        while l < r:
            mid = (l + r) // 2
            
            sum_, counts = 0, 1
            for x in nums:
                sum_ += x
                if sum_ > mid:
                    counts += 1
                    sum_ = x
            
            if counts <= K:
                r = mid
            else:
                l = mid + 1
        
        return l
```

### [4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

```python
# Fav
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        
        # Make sure nums1 longer than nums2, so it is easier
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
            
        if n == 0:
            raise ValueError()
            
        # Partition nums1 and nums2,  all in 2 first parts less than all in 2 second parts
        # Use binary search
        
        # Why +1? if (m + n) is even, 1 doesn't have any effect, if odd 1 will make the middle 
        # in first section
        half_len = (m + n + 1) // 2
        
        # Do binary search on nums1 which is longer, nums2 will be calculated according to nums1
        m_left, m_right = 0, m
        while m_left <= m_right:
            # Must use m_left <= m_right otherwise i never reach to m
            # i is the seperator of nums1 smaller|bigger
            # j should be the num of remaining first half
            # i and j should be first index of second half, cuz index alwasy n-1 
            # i must be [0, m] because there is corner case i == m where all nums1 are smaller
            #   When m == n and nums1[-1] < nums2[0], i should move to right but cannot
            i = (m_left + m_right) // 2
            j = half_len - i
            
            # Move i and j until to the begin/end
            # The corner cases are
            # 1. nums2 are completely smaller or bigger than nums1
            # 2. Based on case1, m == n
            # This corner cases result in i or j to be 0 or end
            
            if i > 0 and j < n and nums1[i - 1] > nums2[j]:
                # If i can move to left and j can move to right and nums1.smaller[-1] still bigger 
                # than nums2.bigger[0]
                m_right = i -1
            elif i < m and j > 0 and nums1[i] < nums2[j - 1]:
                # If i can move to right and j can move to left and nums1.bigger[0] still smaller 
                # than nums2.smaller[-1]
                m_left = i + 1
            else:
                if i == 0:
                    left_max = nums2[j - 1]
                elif j == 0:
                    left_max = nums1[i - 1]
                else:
                    left_max = max(nums1[i - 1], nums2[j - 1])
                    
                if (m + n) % 2 == 1:
                    return left_max
                
                if i == m:
                    right_min = nums2[j]
                elif j == n:
                    right_min = nums1[i]
                else:
                    right_min = min(nums1[i], nums2[j])
                    
                return (left_max + right_min) / 2.0
                
    def findMedianSortedArrays(self, A: List[int], B: List[int]) -> float:
        
        def kth(a, s1, e1, b, s2, e2, k):
            # Intuition: kth() returns the item which is kth smallest in a and b
            # keep reducing the range in a and be util one of it is out of eligible
            # values in kth smallest.
            
            # Trick: Don't use array slicing [i:], it is O(n), instead use
            # start/end indexes
            
            # If s1 doesn't have available items anymore,
            # s1's items in k is exhausted, so items in b within k
            # is k - s1 + 1 the index is k - s1
            if s1 > e1:
                return b[k - s1]
            if s2 > e2:
                return a[k - s2]

            ia, ib = (s1 + e1) // 2, (s2 + e2) // 2
            ma, mb = a[ia], b[ib]

            if ia + ib < k:
                if ma < mb:
                    return kth(a, ia + 1, e1, b, s2, e2, k)
                else:
                    return kth(a, s1, e1, b, ib + 1, e2, k)
            else:
                if ma < mb:
                    return kth(a, s1, e1, b, s2, ib - 1, k)
                else:
                    return kth(a, s1, ia - 1, b, s2, e2, k)

        M, N = len(A), len(B)
        mid = (M + N) // 2

        if (M + N) % 2 == 1:
            return kth(A, 0, M - 1, B, 0, N - 1, mid)
        else:
            return (kth(A, 0, M - 1, B, 0, N - 1, mid) + kth(A, 0, M - 1, B, 0, N - 1, mid - 1)) / 2.0
```



## K Sum



### [167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # Pattern: Binary Search template 2
        N = len(numbers)
        for i, x in enumerate(numbers[:-1]):
            t = target - x
            left, right = i + 1, N - 1
            while left <= right:
                mid = left + (right - left) // 2
                if numbers[mid] == t:
                    return [i + 1, mid + 1]
                elif numbers[mid] > t:
                    right = mid - 1
                else:
                    left = mid + 1
        return []
    
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # Pattern: Two Pointers
        lo, hi = 0, len(numbers) - 1
        while lo <= hi:
            s = numbers[lo] + numbers[hi]
            if s == target:
                return [lo + 1, hi + 1]
            elif s > target:
                hi -= 1
            else:
                lo += 1
```

## Search Matrix

### [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # an empty matrix obviously does not contain `target` (make this check
        # because we want to cache `width` for efficiency's sake)
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        # cache these, as they won't change.
        height = len(matrix)
        width = len(matrix[0])

        # start our "pointer" in the bottom-left
        row = height - 1
        col = 0

        while col < width and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else: # found it
                return True
        
        return False
```
### [1428. Leftmost Column with at Least a One](https://leetcode.com/problems/leftmost-column-with-at-least-a-one/)

```python
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        
        rows, cols = binaryMatrix.dimensions()
        
        # Set pointers to the top-right corner.
        current_row = 0
        current_col = cols - 1
        
        # Repeat the search until it goes off the grid.
        while current_row < rows and current_col >= 0:
            if binaryMatrix.get(current_row, current_col) == 0:
                current_row += 1
            else:
                current_col -= 1
        
        # If we never left the last column, it must have been all 0's.
        return current_col + 1 if current_col != cols - 1 else -1
```

## Others

410. Split Array Largest Sum
1011. Capacity To Ship Packages Within D Days
1231. Divide Chocolate
875. Koko Eating Bananas
774. Minimize Max Distance to Gas Station
1201. Ugly Number III

### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        # Time: O(N * log(Max - Min))
        # Space: O(1)
        N = len(matrix)
        def count_less_equal(x):
            '''
            Given a number x, x doesn't have to exist in matrix. 
            Returns count of numbers in matrix that less or equal
            to x, and also the largest number in matrix <= x (smaller), and
            smallest number in matrix > x (larger).
            Starting from left-bottom, and moving until cannot move. Image,
            this func can draw a line to divide the matrix into to parts, 
            left-top part is nums <= x, and the other half are nums > x. 
            '''
            r, c = N - 1, 0
            # Trick: For sorted matrix, start from left-bottom or right-top 
            # so you can move to smaller or larger by moving r or c
            count, smaller, larger = 0, -math.inf, math.inf
            while r >= 0 and c < N:
                if matrix[r][c] > x:
                    larger = min(larger, matrix[r][c])
                    r -= 1
                else:
                    count += r + 1
                    smaller = max(smaller, matrix[r][c])
                    c += 1
            return count, smaller, larger
        
        # Intuition: Binary search on nums b/w smallest and largest in matrix
        # for each num (mid), it doesnot have to exist in matrix, find how many
        # nums < mid and closet 2 nums around it (smaller and larger). If count == k
        # the smaller is ans (smaller <= mid).
        start, end = matrix[0][0], matrix[-1][-1]
        while start < end:
            mid = (start + end) // 2
            count, smaller, larger = count_less_equal(mid)
            if count == k:
                return smaller
            elif count > k:
                end = smaller
            else:
                start = larger
        return start
```

### [1060. Missing Element in Sorted Array](https://leetcode.com/problems/missing-element-in-sorted-array/)

```python
class Solution:
    def missingElement2(self, nums: List[int], k: int) -> int:
        start = nums[0]
        for i in range(1, len(nums)):
            dist = nums[i] - nums[i - 1] - 1
            if dist < k:
                k -= dist
            else:
                return nums[i - 1] + k
        return nums[-1] + k
    
    def missingElement(self, nums: List[int], k: int) -> int:
        missing = lambda idx: nums[idx] - nums[0] - idx
        N = len(nums)
        
        if missing(N - 1) < k:
            return nums[-1] + k - missing(N - 1)
        
        left, right = 0, N - 1
        # Trick: If right = pivot, use <
        while left < right:
            pivot = left + (right - left) // 2
            if missing(pivot) >= k:
                right = pivot
            else:
                # Trick: Left cannot point to pivot
                left = pivot + 1
        
        # Trick: Left and right will be one finally
        return nums[right-1] + k - missing(right-1)

```

### [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/)

```python
class Solution:
    def singleNonDuplicate1(self, nums: List[int]) -> int:
        left, right = 0, 1
        while right < len(nums):
            if nums[left] != nums[right]:
                return nums[left]
            left += 2
            right += 2
        return nums[-1]

    def singleNonDuplicate(self, nums: List[int]) -> int:
        # Trick: When see sorted array, consider Binart Search
        left, right = 0, len(nums) - 1
        
        while left < right:
            pivot = left + (right - left) // 2
            if pivot % 2 == 1:
                pivot -= 1
            
            if nums[pivot] == nums[pivot + 1]:
                left += 2
            else:
                right = pivot
        return nums[right]
```


### [1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/)

```python
# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        N = mountain_arr.length()
        left, right = 0, N - 1
        while left < right:
            # Pattern: Binary Search
            #   To search a specific value, can use left <= right, but
            #   when the value is not known, use left < right; then mid +/- 1
            #   will change accordingly.
            mid = (left + right) // 2
            # Trick: Check two adjecent item to determin peak
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        peak = left
        
        left, right = 0, peak
        while left <= right:
            mid = (left + right) // 2
            if mountain_arr.get(mid) == target:
                return mid
            elif mountain_arr.get(mid) > target:
                right = mid - 1
            else:
                left = mid + 1
                
        left, right = peak, N - 1
        while left <= right:
            mid = (left + right) // 2
            if mountain_arr.get(mid) == target:
                return mid
            elif mountain_arr.get(mid) < target:
                right = mid - 1
            else:
                left = mid + 1
                
        return -1
```

### [852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        N = len(arr)
        left, right = 0, N - 1
        while left < right:
            mid = (left + right) // 2
            # Cannot use arr[mid - 1] < arr[mid], cuz it cannot
            # prov target is at right of mid, it could be mid;
            # left has to be mid + 1, otherwise dead loop
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left
```

### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        
        m, n = len(matrix), len(matrix[0])
        
        left, right = 0, m * n - 1
        
        # If both left and right move +/-1, use left <= right
        while left <= right:
            mid = (left + right) // 2
            # Trick: Map a index to matrix coordinatts
            v = matrix[mid // n][mid % n]
            if target == v:
                return True
            elif target < v:
                right = mid - 1
            else:
                left = mid + 1
        return False
```

### [528. Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/)

```python
class Solution:

    def __init__(self, w: List[int]):
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum

    def pickIndex(self) -> int:
        target = self.total_sum * random.random()
        
        low, high = 0, len(self.prefix_sums) - 1
        while low < high:
            mid = low + (high - low) // 2
            if target > self.prefix_sums[mid]:
                low = mid + 1
            else:
                high = mid
        return low
            


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
```

### [704. Binary Search](https://leetcode.com/problems/binary-search/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

### [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)

```python
class Solution:
    def smallestDivisor(self, A, threshold):
        l, r = 1, max(A)
        while l < r:
            m = (l + r) // 2
            # math.ceil(i/m)
            if sum(math.ceil(i/m) for i in A) > threshold:
                l = m + 1
            else:
                r = m
        return l
```