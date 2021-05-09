
### [50. Pow(x, n)](https://leetcode.com/problems/powx-n/)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def my_pow(x, n):
            if n == 0:
                return 1.0
            
            half = my_pow(x, n//2)
            
            ans = half * half
            
            if n%2 == 1:
                ans = ans * x
                
            return ans
        
        if n < 0:
            n = -n
            x = 1/x
            
        return my_pow(x, n)
```

### [241. Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)

```python
class Solution:
    #Trick: String parsing for math expression
    ops = {
        '+': lambda a, b: str(int(a) + int(b)),
        '-': lambda a, b: str(int(a) - int(b)),
        '*': lambda a, b: str(int(a) * int(b))
    }
    def diffWaysToCompute(self, input: str) -> List[int]:
        # Trick: Divide and Conquer
        #    Another resursion method, difference with others is it do own
        #    operation after the sub problem call. It depends on where is 
        #    the complexity, then use resursion to get rid of it.
        ans = []
        for i in range(len(input)):
            ch = input[i]
            if ch in '+-*':
                ans_l = self.diffWaysToCompute(input[:i])
                ans_r = self.diffWaysToCompute(input[i + 1:])
                # This is faster
                # for l in ans_l:
                #     for r in ans_r:
                #         ans.append(self.ops[ch](l, r))
                ans.extend([self.ops[ch](l, r) for l in ans_l for r in ans_r])
        return ans if ans else [input]
```

### [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

```python
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def dist(i):
            return points[i][0] ** 2 + points[i][1] ** 2
        
        # https://www.youtube.com/watch?v=G9VcMTSZ1Lo&t=18s
        def partition(i, j):
            oi = i
            pivot = dist(i)
            i += 1
            
            while True:
                # Find the leftmost value that > pivot
                while i < j and dist(i) < pivot:
                    i += 1
                # Find the rightmost value < pivot
                while i <= j and dist(j) >= pivot:
                    j -= 1
                # If pointers meet, complete
                if i >= j: break
                # If pointers not meet, swap
                points[i], points[j] = points[j], points[i]
                
            # j is currently the last value < pivot
            points[oi], points[j] = points[j], points[oi]
            
            # After final swap, j become mid
            return j

    
        def sort(i, j, K):
            if i >= j: return

            mid = partition(i, j)

            len_left = mid - i + 1
            if len_left > K:
                sort(i, mid - 1, K)
            elif len_left < K:
                sort(mid + 1, j, K - len_left)

        sort(0, len(points) - 1, K)
        return points[:K]
```

