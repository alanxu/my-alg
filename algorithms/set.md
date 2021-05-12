### [349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return set(nums1) & set(nums2)
```

### [350. Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        counter1, counter2 = Counter(nums1), Counter(nums2)
        if len(counter1) > len(counter2):
            counter1, counter2 = counter2, counter1

        ans = []
        
        for n, c1 in counter1.items():
            if n in counter2:
                c2 = counter2[n]
                ans.extend([n for _ in range(min(c1, c2))])
        
        return ans
    
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        counter1, counter2 = Counter(nums1), Counter(nums2)
        if len(counter1) > len(counter2):
            counter1, counter2 = counter2, counter1
        
        # Trick: Use nums1 to reduce space to O(1)
        k = 0
        for n, c1 in counter1.items():
            if n in counter2:
                c2 = counter2[n]
                l = min(c1, c2)
                nums1[k:k + l] = [n for _ in range(l)]
                k += l
        
        return nums1[:k]
```