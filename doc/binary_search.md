#

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