
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

