#

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