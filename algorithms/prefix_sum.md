# Prefix Sum


### [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        # Trick: Prefix Sum
        count, ans = 0, 0
        # If presum is 0, all nums so far matches
        sums = {0: -1}
        for i in range(len(nums)):
            count += 1 if nums[i] == 1 else -1
            if count in sums:
                ans = max(ans, i - sums[count])
            else:
                sums[count] = i
        return ans
```