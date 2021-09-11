### [398. Random Pick Index](https://leetcode.com/problems/random-pick-index/)

```python
class Solution:

    def __init__(self, nums: List[int]):
        self.ids = defaultdict(list)
        for i, x in enumerate(nums):
            self.ids[x].append(i)

    def pick(self, target: int) -> int:
        return random.choice(self.ids[target])


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)
```