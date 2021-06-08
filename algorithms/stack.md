

1130. Minimum Cost Tree From Leaf Values
907. Sum of Subarray Minimums
901. Online Stock Span
856. Score of Parentheses
503. Next Greater Element II
496. Next Greater Element I
84. Largest Rectangle in Histogram
42. Trapping Rain Water


## Parenthsese

### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        # Edge cases: Odd-lenthed s, '((', '))'
        # Intuition: Push every left quaots, for
        # each right quaot, return False if stack
        # is empty (no match left), return False if
        # there is left but not same type.
        # Finally check if stack is exausted to avoid
        # '((' case.
        if len(s) % 2: return False
        stack = []
        pairs = {'{': '}', '[': ']', '(': ')'}
        for x in s:
            if x in pairs:
                stack.append(x)
            elif not stack or pairs[stack.pop()] != x:
                    return False
        return not stack
```

### [1249. Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack, result = [], ''
        for i, x in enumerate(s):
            if x == '(':
                stack.append(i)
                result += x
            elif x == ')':
                if stack:
                    stack.pop()
                    result += x
                else:
                    result += '*'
            else:
                result += x
        
        # Trick: Cast str to list to replace a char
        result = list(result)
        for i in stack:
            # In stack is the index
            result[i] = '*'
        
        return ''.join(x for x in result if x != '*')
```

### [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # Intuition: Stack - Indexes not in stack are
        # part of a valid str
        #
        # Iterate s:
        # If '(' always
        # in stack; 
        #
        # If ')' at k, can match '(' in stack top at j,
        # the valid parenthess substr len is (k - i), 
        # because there might be valid ans between i and j
        # that removed
        # x x (        )
        #   i j        k
        #
        # If ')' no matched '(' at stack top, ')' in stack,
        # and it will never be popped up, it becomes a anchor
        # for (k - i)
        
        # Trick: Dummy entry for no entry in stack (all str are
        # valid)
        stack = [(-1, ')')]
        ans = 0
        for i, x in enumerate(s):
            if x == ')' and stack[-1][1] == '(':
                stack.pop()
                ans = max(ans, i - stack[-1][0])
            else:
                stack.append((i, x))
        return ans
```

### [636. Exclusive Time of Functions](https://leetcode.com/problems/exclusive-time-of-functions/)

```python
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        stack = []
        time = [0] * n
        for log in logs:
            log_segs = log.split(':')
            fun_id, status, ts = int(log_segs[0]), log_segs[1], int(log_segs[2])
            #print((fun_id, status, ts))
            if status == 'start':
                stack.append((fun_id, ts))
            elif status == 'end':
                _fun_id, _ts = stack.pop()
                # print((_fun_id, ts, _ts))
                # We know if cur status is 'end', the stack top
                # is always the matching 'start', cuz we pop everytime for an 'end'
                elapse_time = ts - _ts + 1
                time[_fun_id] += elapse_time
                
                if stack:
                    # If stack is not empty, the current func has a parent func,
                    # the parent func's exclusive time needs to minus the child func.
                    # Inclusive time can be found from log, so no need worry calculate outter level
                    # The parent func don't need to be poped, cuz you don't know when it ends atm
                    # but what you are sure is the parent func's time must minus the current func time
                    # so you just deduct it so it counts when parent func inclusive time is calculated
                    time[stack[-1][0]] -= elapse_time
        return time
```

### [856. Score of Parentheses](https://leetcode.com/problems/score-of-parentheses/)

```python
class Solution(object):
    def scoreOfParentheses(self, S):
        # Intuition: The whole structure is a forest.
        # (if there is a parentheses that capture everything 
        # then it's a tree). Essentially we're calculating the 
        # sum of leaves. For each leave, the weight is 2^(depth-1)
        ans = bal = 0
        for i, x in enumerate(S):
            # Trick: Use balance to track depth of quotes
            if x == '(':
                bal += 1
            else:
                bal -= 1
                if S[i - 1] == '(':
                    ans += 1 << bal
        return ans
    
    def scoreOfParentheses(self, S):
        # Intuition: Still image this is a forest.
        # Use stack to trace score of each depth.
        # When '(', a new depth/branch is started, put 
        # 0 first, when ')' a branch is closed, pop
        # it and handle next
        stack = [0]
        for x in S:
            if x == '(':
                stack.append(0)
            else:
                v = stack.pop()
                stack[-1] += max(2 * v, 1)
        return stack.pop()
```

### [1190. Reverse Substrings Between Each Pair of Parentheses](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        stack = []
        for x in s:
            if x!=")": #push everything into stack except ")"
                stack.append(x)
            else: #if we meet ")", pop all the letters until we meet "("
                new_s = ""
                while(stack):
                    last = stack.pop()
                    if last == "(":
                        break
                    new_s += last[::-1]
                stack.append(new_s) #append the reverse substring
        return "".join(stack)
```

### [726. Number of Atoms](https://leetcode.com/problems/number-of-atoms/)

```python
class Solution:
    def countOfAtoms(self, f: str) -> str:
        N = len(f)
        # Trick: Use Counter for each level
        stack = [collections.Counter()]
        i = 0
        # Trick: Use while and len based index for flexible control
        while i < N:
            if f[i] == '(':
                stack.append(collections.Counter())
                i += 1
            elif f[i] == ')':
                i += 1
                i_start = i
                while i < N and f[i].isdigit(): i += 1
                quantity = int(f[i_start:i] or 1)
                # If cur is ')', look for quantity then, the quantity
                # is applied for the whole thing in (), and pop the top
                # () and update outter using top's data and quantity.
                top = stack.pop()
                for name, v in top.items():
                    stack[-1][name] += v * quantity
            else:
                # If cur is letter or number, scan name and quantity 
                # seperately and updat cur stack top
                i_start = i
                i += 1
                while i < N and f[i].islower(): i += 1
                name = f[i_start:i]
                i_start = i
                while i < N and f[i].isdigit(): i += 1
                quantity = int(f[i_start:i] or 1)
                stack[-1][name] += quantity
                
        return "".join(name + (str(stack[-1][name] if stack[-1][name] > 1 else '')) for name in sorted(stack[-1]))
    
    def countOfAtoms(self, formula):
        parse = re.findall(r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)", formula)
        stack = [collections.Counter()]
        for name, m1, left_open, right_open, m2 in parse:
            print((name, m1, left_open, right_open, m2))
            if name:
              stack[-1][name] += int(m1 or 1)
            if left_open:
              stack.append(collections.Counter())
            if right_open:
                top = stack.pop()
                for k in top:
                  stack[-1][k] += top[k] * int(m2 or 1)

        return "".join(name + (str(stack[-1][name]) if stack[-1][name] > 1 else '')
                       for name in sorted(stack[-1]))
```


## Iterator

### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self.push(root)

    def next(self) -> int:
        node = self.stack.pop()
        # If node has right, it is
        # a root of that sub tree,
        # In order requires right to 
        # be processed after root, so
        # push node.right;
        # If node doesnt have right,
        # it means its subtree is 
        # complete
        if node.right:
            self.push(node.right)
        return node.val

    def hasNext(self) -> bool:
        return not not self.stack
    
    def push(self, node):
        # Recursively push root and left
        # the most left node is on top
        self.stack.append(node)
        if node.left:
            self.push(node.left)
```

### [341. Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/)

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # Use reversed() because we want first
        # element in stack top
        self.stack = list(reversed(nestedList))
    
    def next(self) -> int:
        self.flaten_stack_top()
        return self.stack.pop()
    
    def hasNext(self) -> bool:
        self.flaten_stack_top()
        return len(self.stack) > 0
        
    def flaten_stack_top(self) -> None:
        # Use while so stack top is guaranteed to be int
        while self.stack and not self.stack[-1].isInteger():
            self.stack.extend(reversed(self.stack.pop().getList()))
```

## Monotonic Stack

### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

```python
# Fav
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Intuition: Monotonic Stack
        # For each bar index x, maintain a monotonic increase stack 
        # with x as stack top. 
        # Before push x to stack, check the stack
        # top y, if height of y > x, y needs to be popped. Then we
        # got x, y and cur stack top z. We know that all bars btw
        # z and y are higher than y and y/x should be adjecent.
        # So if heights[y] > heights[x], it means a max rectangle 
        # determins by y can be calculated, the width is [z + 1:x - 1]
        # there is corner case that y is only elements, meaning y
        # is smaller than all bars before it, then width is [0:x - 1]
        # After calculated y in case of height[y] > height[x], we can
        # push x to stack, if otherwise height[y] <= height[x], then
        # y and x are both in stack and wait for a smaller bar in future
        # to clear both.
        # Time: O(n) - n bars pushed and poped
        # Space: O(n)
        ans, stack = 0, []
        # Trick: Dummy 0-heighted bar to clear all remaining bars if any
        heights.append(0)
        for i, x in enumerate(heights):
            # Trick: Use index in mono stack cuz height can be
            # obtained from heights array.
            # Note using while not if
            while stack and x < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1 if stack else i
                ans = max(ans, w * h)
            stack.append(i)
        return ans
```

### [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # Intuition: For each row, treat it as ground, each column
        # are bars. For each row calc the max rectangle, and get max
        # using #84
        def _84(heights):
            ans, stack = 0, []
            heights.append(0)
            for i, x in enumerate(heights):
                while stack and x < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i - stack[-1] - 1 if stack else i
                    ans = max(ans, h * w)
                stack.append(i)
            return ans
        
        if not matrix:
            return 0
        ans = 0
        heights = [0] * len(matrix[0])
        for row in matrix:
            for i in range(len(row)):
                # Trick: if-else is greedy
                heights[i] = heights[i] + 1 if row[i] == '1' else 0
            ans = max(ans, _84(heights))  
        return ans
```

### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # Intuition: Monotonic Decreasing Stack
        # Calc ans for each item when popped out from stack. A item j
        # is popped out when there is another item after j greater than j.
        # It means j has an non-zero answer. All items remaining in stack
        # when loop finish dont have an i greater than them, so ans for
        # them is 0.
        ans, stack = [0] * len(temperatures), []
        for i, x in enumerate(temperatures):
            while stack and x > temperatures[stack[-1]]:
                j = stack.pop()
                ans[j] = i - j
            stack.append(i)
        return ans
```

### [402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        
        # Intuition: The num is smallest when keep a mono increase
        # stack and remove others. If k is small, make more significant
        # digits mono increase, else if k is big enough to make all
        # digits mono increase and has remaining, use the remaining to
        # remove from tail, cuz it is mono increasing.
        
        # Construct a monotonic increasing sequency
        # starting from left. Simple append remaining numbers
        # once k deletion is already made.
        for d in num:
            while k and stack and stack[-1] > d:
                stack.pop()
                k -= 1
                
            stack.append(d)
        # If k is exausted, it yeild the num_stack with the significant
        # digits handled as much as possible with k deletions, and less 
        # significant parts remains unchanged so it is the final minimum mumber;
        # If k is not exausted, it means the whole num is
        # made monotonic increasing with <k deletions, so just 
        # remove remaining deletion quotas (<k)
        stack = stack[:-k] if k else stack
        
        # Trick: lstrip('0')
        return ''.join(stack).lstrip('0') or '0'
```

### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

```python
# Fav
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # Intuition: Duplicate the nums, then it becomes 739.
        # Higer Tempreture. Only return first half of ans.
        N = len(nums)
        nums, stack, ans = nums * 2, [], [-1] * N * 2
        for i, x in enumerate(nums):
            while stack and nums[stack[-1]] < x:
                j = stack.pop()
                ans[j] = x
            stack.append(i)
        return ans[:N]
    
    def nextGreaterElements(self, A):
        # Intuition: Improve time and space by doing
        # two loops rather than duplicate array.
        stack = []
        ans = [-1] * len(A)
        
        for i, a in enumerate(A):
            while stack and a > A[stack[-1]]:
                ans[stack.pop()] = a
            stack.append(i)
            
        for i, a in enumerate(A):
            while stack and a > A[stack[-1]]:
                ans[stack.pop()] = a
            # Trick: Don't push in 2nd loop, only use 2nd
            # loop to pop 1st loop in stack.
            if not stack:
                break
        
        return ans
```

### [581. Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # Doesn't work
        stack, start, end = [], math.inf, math.inf
        for i, x in enumerate(nums):
            while stack and nums[stack[-1]] >= x:
                start, end = min(start, stack.pop()), i
            stack.append(i)
        return end - start + 1 if start != math.inf else 0
    
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # Intuition: Mono Increasing Stack.
        # While constructing mono increaseing stack, 
        # If the array is perfectly sorted ascendingly, no
        # item will be popped. On the other hand, if a
        # item is popped, it means the item needs to be resorted.
        # If we can find the beging of the popped item as start, and
        # track the last item that squeez others out as end, we
        # can resolve the problem.
        # But, above solution doesn't work for cases with duplication.
        # 1 3 2 2 2
        # In this case, multiple 2s won't squeeze each other, so
        # we cannot find correct end.
        # But we know we can get correct start, so we just need to
        # do the process 2nd time with reversed order to find the
        # end.
        stack, start, end = [], math.inf, -math.inf
        
        for i, x in enumerate(nums):
            while stack and nums[stack[-1]] > x:
                start = min(start, stack.pop())
            stack.append(i)
        
        stack = []
        for i, x in reversed(list(enumerate(nums))):
            while stack and nums[stack[-1]] < x:
                end = max(end, stack.pop())
            stack.append(i)
        
        return 0 if start > end else end - start + 1
    
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # Intuition: Sort nums and compare with original one,
        # get the first and last pos that the number doesn't same.
        ans = [i for i, (a, b) in enumerate(zip(nums, sorted(nums))) if a != b]
        return ans[-1] - ans[0] + 1 if ans else 0
            
```

### [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/)

```python
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        # dp is the max score can achieve at step i
        N = len(nums)
        dp = [0] * N
        dp[0] = nums[0]
        
        # q is monotonicc decreasing queue for prev best scores for step
        # [i - k, i - 1]
        q = deque([0])
        for i in range(1, N):
            # For each step, get rid of pre step cannot be reached back
            if q[0] < i - k:
                q.popleft()
            
            # The head of monotonic decreasing stack/queue is max score
            # dp[i] = max(dp[i - k],...,dp[i - 1]) + nums[i]
            # max is q[0]
            dp[i] = dp[q[0]] + nums[i]
            
            # Update the mono queue using current dp[i]
            # > and >= both work
            while q and dp[i] > dp[q[-1]]:
                q.pop()
            q.append(i)
        
        return dp[-1]
```

### [155. Min Stack](https://leetcode.com/problems/min-stack/)

```python
class MinStack:
    # Intuition: Monotonic Stack
    # Use a monotonic decreasing stack to track
    # the desc seq in the stack, when the stack
    # top eques mono stack top in pop, pop both
    # otherwise it means mono stack top is not
    # reached in stack;
    # When push, onl push to mono stack when
    # val is a new min
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = [math.inf]
        self.mono_desc_stack = [math.inf]

    def push(self, val: int) -> None:
        self.stack.append(val)
        # <= is important to handle same val
        if val <= self.mono_desc_stack[-1]:
            self.mono_desc_stack.append(val)

    def pop(self) -> None:
        x = self.stack.pop()
        if x == self.mono_desc_stack[-1]:
            self.mono_desc_stack.pop()

    def top(self) -> int:
        x = self.stack[-1]
        return x if x != math.inf else None

    def getMin(self) -> int:
        x = self.mono_desc_stack[-1]
        return x if x != math.inf else None


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

### [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)
```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        
        for i in range(len(s)):
            # Why ignore s[i] when the value already used? - It will never generate 
            # more optimal solotion. It will never supass previous same value - If
            # prev same value cannot pop values in from it, neither the cur value;
            # and it will not help to get optimal value after previous same value, values
            # after s[i] will keep optimze remaining:
            # Consider "abczcyz", 2nd c is ignored, but y can pop first z, so leave it
            # for capable people, dont do everything yourself.
            if s[i] not in stack:
                while stack and stack[-1] > s[i] and stack[-1] in s[i:]:
                    stack.pop()
                stack.append(s[i])
            
        return ''.join(stack)
```


## Others




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

### [71. Simplify Path](https://leetcode.com/problems/simplify-path/)

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        path_segments = path.split('/')
        stack = []
        
        for seg in path_segments:
            if not seg or seg == '.':
                continue
            elif seg == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(seg)
                
        return '/' + '/'.join(stack)
```


### [224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)

```python
class Solution:
    def calculate(self, s: str) -> int:
        
        def update(op, v):
            if op == "+": stack.append(v)
            if op == "-": stack.append(-v)
            if op == "*": stack.append(stack.pop() * v)
            if op == "/": stack.append(int(stack.pop() / v))

        num, sign = 0, '+'
        stack = []
        idx = 0
        
        while idx < len(s):
            c = s[idx]
            
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in '+-*/':
                update(sign, num)
                num, sign = 0, c
            elif c == '(':
                num, j = self.calculate(s[idx + 1:])
                idx += j
            elif c == ')':
                update(sign, num)
                print(stack)
                return sum(stack), idx + 1
                
            idx += 1

        update(sign, num)

        return sum(stack)
```

### [227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)

```python
class Solution:
    def calculate(self, s: str) -> int:
        
        def update(op, v):
            if op == "+": stack.append(v)
            if op == "-": stack.append(-v)
            if op == "*": stack.append(stack.pop() * v)
            if op == "/": stack.append(int(stack.pop() / v))

        num, sign = 0, '+'
        stack = []
        
        for idx in range(len(s)):
            c = s[idx]
            
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in '+-*/':
                update(sign, num)
                num, sign = 0, c
        print(stack)
        update(sign, num)
        
        return sum(stack)
```

### [772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)

```python
class Solution:
    def calculate(self, s: str):
        
        def update(op, v):
            if op == "+": stack.append(v)
            if op == "-": stack.append(-v)
            if op == "*": stack.append(stack.pop() * v)
            if op == "/": stack.append(int(stack.pop() / v))

        num, sign = 0, '+'
        stack = []
        idx = 0
        
        while idx < len(s):
            c = s[idx]
            
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in '+-*/':
                update(sign, num)
                num, sign = 0, c
            elif c == '(':
                num, j = self.calculate(s[idx + 1:])
                idx += j
            elif c == ')':
                update(sign, num)
                print(stack)
                return sum(stack), idx + 1
                
            idx += 1

        update(sign, num)

        return sum(stack)
```

### [394. Decode String](https://leetcode.com/problems/decode-string/)

```python
class Solution:
    def decodeString(self, s: str) -> str:
        number_stack, str_stack = [], []
        
        # Current str is result, of current layer, it bump to last/next level according to []
        k_str, cur_str = '', ''
        for c in s:
            if c.isdigit():
                k_str += c
            elif c == '[':
                number_stack.append(int(k_str))
                str_stack.append(cur_str)
                k_str, cur_str = '', ''
            elif c == ']':
                decoded_str = str_stack.pop()
                k = number_stack.pop()
                cur_str = decoded_str + cur_str * k
            else:
                cur_str += c
        return cur_str
```

### [735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision/)

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for new in asteroids:
            explod = False
            # Trick: Check negative and positive numbers 
            while stack and new < 0 < stack[-1]:
                if stack[-1] > -new:
                    explod = True
                    break
                elif stack[-1] == -new:
                    explod = True
                    stack.pop()
                    break
                elif stack[-1] < -new:
                    stack.pop()
            
            if not explod:
                stack.append(new)
                
        return stack
```

### [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)

```python
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span
        


# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)
```

### [1209. Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

```python
class Solution:
    def removeDuplicates(self, s, k):
        stack = [['#', 0]]
        for c in s:
            if stack[-1][0] == c:
                stack[-1][1] += 1
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append([c, 1])
        return ''.join(c * k for c, k in stack)
```

### [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

```python
class MyStack:
    # Intuition: On push call, following push of
    # the cur value, pop and push previous
    # values L - 1 times, so the queue is
    # NEW OLD1 OLD2 OLD3 ...
    def __init__(self):
        self.q = deque()
        self.l = 0

    def push(self, x: int) -> None:
        self.q.append(x)
        self.l += 1
        for _ in range(self.l - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        if self.l > 0: self.l -= 1
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return self.l == 0
```

### [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

```python
class MyQueue:

    def __init__(self):
        self.stack1, self.stack2 = [], []
        
    # Solution 1: When push every value, use
    # two stack to make sure latest always
    # at bottom.
    # push: O(n), pop: O(1)
    def push(self, x: int) -> None:
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        self.stack1.append(x)
        while self.stack2:
            self.stack1.append(self.stack2.pop())
        
    def pop(self) -> int:
        return self.stack1.pop()

    def peek(self) -> int:
        return self.stack1[-1]
        
    def empty(self) -> bool:
        return not self.stack1
    
    # Solution 2: Use 2 stack to reverse order
    # of elements. This works only when s2 is empty
    # Use s1 as a cache, s2 is the primary.
    # When peek(), check if primary is empty, if not
    # just use primary, if empty, copy from s1 in reversed
    # order. When push, just push to s1
    # push: O(1), pop: amortized O(1)
    
    # Alg: amortized O(1) means most of call is O(1), but
    # sometime could be O(n), e.g. ArrayList resize
    def push(self, x: int) -> None:
        self.stack1.append(x)
        
    def pop(self) -> int:
        self.peek()
        return self.stack2.pop()

    def peek(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
        
    def empty(self) -> bool:
        return not self.stack1 and not self.stack2
```

### [456. 132 Pattern](https://leetcode.com/problems/132-pattern/submissions/)

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        N = len(nums)
        
        # Iterate array to calc min value at left of i
        # including i
        left_min = [nums[0]] + [-1] * (N - 1)
        for i in range(1, N):
            left_min[i] = min(left_min[i - 1], nums[i])
        
        # Reversely traverse array, for every x
        stack = []
        for x in range(N - 1, 0, -1):
            # Examin if x has a valid i, if
            # not x cannot by j, and because
            # x < left_min[x], x < any left_min[j]
            # where 0 <= j < x.
            # So x cannot be j, k, so dump it
            # and x cannot be i, cuz if it is
            # a valid i, the programme has returned
            # True.
            if nums[x] <= left_min[x]:
                continue
            
            # If x has valid i, examin if x could be j
            while stack and stack[-1] <= left_min[x]:
                # Check the stack from top, if the top
                # is smaller than smallest at left of x,
                # it cannot be x's k, and it cannot be k
                # for any j at left of x, so dmp it.
                # And it cannot be i and j otherwise it 
                # should not be here.
                stack.pop()
            
            # After popping out all too-small k, check
            # if k < j, if so return True.
            # Why just check top? If stack[-1] >= nums[x], 
            # why x cannot be j? Because the stack is
            # actually a monotonic increas stack, top
            # is the smallest in stack. So if first one
            # is too big, all others is too big too.
            if stack and stack[-1] < nums[x]:
                return True
            
            # If x cannot be i and j, maybe it can
            # be k, so push to stack.
            # Why use Stack? Because we have to check top
            # in monotonic increasing order. Why it is
            # monotonic increasing? First, starting from
            # right, all x in stack is not the smallest
            # of his left (inclusive) which means it must
            # be smallest on it's right, otherwise it already
            # return True, so each the stack is in monotonic
            # order, and only Stack can maintain this and
            # process in correct order.
            stack.append(nums[x])
        return False
```

### [946. Validate Stack Sequences](https://leetcode.com/problems/validate-stack-sequences/)

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # Intuition: Greedy - Push nums in order, after push each x,
        # try pop if stack top matched the num to be popped, for each
        # push, needs a loop to keep checking pop until not matching.
        stack = []
        for x in pushed:
            stack.append(x)
            while stack and stack[-1] == popped[0]:
                stack.pop()
                popped.pop(0)

        return not popped
```