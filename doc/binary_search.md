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
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[right]:
                right = mid
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


## Others

378. Kth Smallest Element in a Sorted Matrix
410. Split Array Largest Sum
1011. Capacity To Ship Packages Within D Days
1231. Divide Chocolate
875. Koko Eating Bananas
774. Minimize Max Distance to Gas Station
1201. Ugly Number III

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