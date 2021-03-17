
### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(left, right, pivot_index):
            pivot = nums[pivot_index]
            # 1. Move pivot to right
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            # 2. Move all smaller than pivot to the left
            write = left
            for read in range(left, right):
                if nums[read] < pivot:
                    nums[write], nums[read] = nums[read], nums[write]
                    write += 1
            # 3. When read has scanned all num except pivot at very right, put pivot
            #    at the write location
            nums[write], nums[right] = nums[right], nums[write]
            
            return write
            
        def select(left, right, k_smallest):
            if left == right:
                return nums[left]
            # 1. Gen a random pivot index
            pivot_index = random.randint(left, right)
            # 2. Find the pivot position as if in sorted list
            pivot_index = partition(left, right, pivot_index)
            # 3. Check the pivot index
            if pivot_index == k_smallest:
                return nums[k_smallest]
            elif pivot_index < k_smallest:
                # Why no need to check edge case? If pivot_idx < k_small,
                # there must be num greater than it, so pivot_idx + 1 must
                # be valid
                return select(pivot_index + 1, right, k_smallest)
            else:
                return select(left, pivot_index - 1, k_smallest)
            
        return select(0, len(nums) - 1, len(nums) - k)
```