

## Problems

### [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)
```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        
        for i in range(len(s)):
            # Why ignore s[i] when the value already used? - It will never generate 
            # more optimal solotion. It will never supass previous same value, and
            # it will not help to get optimal value after previous same value, values
            # after s[i] will keep optimze remaining
            if s[i] not in stack:
                while stack and stack[-1] > s[i] and stack[-1] in s[i:]:
                    stack.pop()
                stack.append(s[i])
            
        return ''.join(stack)
```


### [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)

```python
# Fav
class Solution:
    def sumSubarrayMins(self, A):
        # Trick: Monotonic Stack
        #     Monotone Stack maintains the Previous Less Element of the current position in array.
        #     It is the increasing elements from begining till cur pos.
        #     It usually store the index of the elements, all other elements between two indexes in
        #     the stack are greater than both.
        
        # Trick: Dummy items
        #     First dummy is to handle edge case when only one element in stack for stack[-1]
        #     Second dummy is make sure all elements is processed, as well as edge case of (i - cur)
        A = [-math.inf] + A + [-math.inf]
        N, stack, ans = len(A), [], 0
        
        # For each element in array, calculate the number of subarray with A[i] as min;
        #   For example:
        #     in [9,7,8,3,4,6]
        #     we have 4 choices to start with (9,7,8,3)
        #     we have 3 choices to end with (3,4,6)
        #     So answer is just 4*3.
        for i in range(N):
            while stack and A[stack[-1]] > A[i]:
                # When pop an element, it is time to calculate with the popped element as min
                # stack[-1], cur, i forms the scope of all possible subarray for cur, with stack[-1]
                # and i exclusive.
                # This way also magically resolve edge cases when there are multiple same min value
                # in the array. Same value will be stacked, then all same value are popped by
                # next less element or the end -inf, last same value popped will cover the case that
                # all same min in same subarray by doing (i - cur)
                cur = stack.pop()
                ans += A[cur] * (i - cur) * (cur - stack[-1])
            stack.append(i)
            
        return ans % (10 ** 9 + 7)
```


### [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        
        operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: int(a / b)
        }
        
        for s in tokens:
            if s in operations:
                n2, n1 = stack.pop(), stack.pop()
                stack.append(operations[s](n1, n2))
            else:
                stack.append(int(s))
        return stack[-1]
```

## Iterator

### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self.push(root)

    def next(self) -> int:
        if self.stack:
            node = self.stack.pop()
            if node.right:
                self.push(node.right)
            return node.val

    def hasNext(self) -> bool:
        if self.stack:
            return True
        return False
        
    def push(self, node):
        if node:
            self.stack.append(node)
            if node.left:
                self.push(node.left)
```

### [341. Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/)

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = list(reversed(nestedList))
    
    def next(self) -> int:
        self.make_stack_top_a_integer()
        return self.stack.pop().getInteger()
        
    
    def hasNext(self) -> bool:
        self.make_stack_top_a_integer()
        return len(self.stack) > 0
         
    def make_stack_top_a_integer(self):
        while self.stack and not self.stack[-1].isInteger():
            self.stack.extend(reversed(self.stack.pop().getList()))
```