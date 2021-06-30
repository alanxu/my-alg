
### [306. Additive Number](https://leetcode.com/problems/additive-number/)

```python
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        N = len(num)
        def backtrack(pre1='0', pre2='0', i=0, n=0):
            if i == N:
                return True
            
            if pre1[0] == '0' and pre1 != '0' or pre2[0] == '0' and pre2 != '0':
                return False
            
            if n == 0:
                # If process first number, i = 0, the max posibility of j is N // 2
                for j in range(1, N // 2 + 1):
                    if backtrack(num[:j], '0', j, n + 1):
                        return True
            elif n == 1:
                # If process second number, i is start of second, j is in scope of below
                for j in range(i + 1, i + (N - i) // 2 + 1):
                    if backtrack(pre1, num[i:j], j, n + 1):
                        return True
            else:
                # If process third number, i is starting of third, cut third by length
                cur = str(int(pre1) + int(pre2))
                j = i + len(cur)
                if cur == num[i:j] and backtrack(pre2, cur, j, n + 1):
                    return True
            return False
        
        return backtrack()
```

### [1239. Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/)

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        # Trick: Back Tracking
        # Use a dfs alg to generate complete combination of N items;
        # The complete tree is a unbalanced tree, left-most child is deepest in all level;
        # The full combination is the very left leaf, the last item is the very right leaf;
        # The backtrack() function has change to access all the comb, so rules can be applied
        # based on questions;
        # Sometimes question is focused on combination of X items, sometimes num of items doesn't
        # matter, all are using save way which is searching same tree;
        # Before visite children, there should be end condition, it doesn't has to be very first line,
        # but has to be before visit children;
        # The algo typically use left to right fashion, using an index (start) to mark the left boarder
        # of the current func;
        # Each func is taking an node as root, and use dfs for each of its children;
        # The key of performance is pruning, dont process the children if the parent is not matching
        N = len(arr)
        self.ans = 0
        def backtrack(s='', start=0):
            # Trick: Pruning
            if len(s) != len(set(s)): return
            self.ans = max(self.ans, len(s))
            if start == N: return
            for i in range(start, N):
                backtrack(s + arr[i], i + 1)
            
        backtrack()
        return self.ans
    
    def maxLength1(self, A):
        dp = [set()]
        for a in A:
            # Trick: Use set() to check string duplicate
            if len(set(a)) < len(a): continue
            a = set(a)
            for c in dp[:]:
                # Trick: Use set() to manipulate dupicated string
                if a & c: continue
                dp.append(a | c)
        return max(len(a) for a in dp)
```


### [351. Android Unlock Patterns](https://leetcode.com/problems/android-unlock-patterns/)

```python
# Fav
class Solution:
    def numberOfPatterns(self, m: int, n: int) -> int:
        # Trick: Back Tracking vs. DFS
        #   They are different. DFS is used to travrse a tree or graph without duplication;
        #   while Back Tracking is to build and traverse a tree derived from some rule on another data structure
        #   (combination of lists, traverse order of a graph);
        #   Because the combination feature, it require change/rollback each step, but DFS doesn't need
        has_obstacle = {
            (1,3): 2, (3,1): 2,
            (1,7): 4, (7,1): 4,
            (3,9): 6, (9,3): 6,
            (7,9): 8, (9,7): 8,
            (1,9): 5, (9,1): 5,
            (2,8): 5, (8,2): 5,
            (3,7): 5, (7,3): 5,
            (4,6): 5, (6,4): 5
        }
        
        self.ans = 0
        def backtrack(num, step):
            if m <= step <= n:
                self.ans += 1
            if step == n:
                return
            # Mark visited if not reach n
            visited.add(num)
            for nxt in range(1, 9 + 1):
                if nxt not in visited:
                    if (num, nxt) in has_obstacle and has_obstacle[(num, nxt)] not in visited:
                        continue
                    backtrack(nxt, step + 1)
            # REMOVE it for back tracking
            visited.remove(num)
        
        for i in range(1, 9 + 1):
            visited = set()
            backtrack(i, 1)
        
        return self.ans

```

### [1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])

        self.ans = 0
        def backtrack(r, c, cur_amount=0, visited=None):
            if not visited:
                visited = set()
            cur_amount += grid[r][c]
            visited.add((r, c))
            is_end = True
            for _r, _c in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] and (_r, _c) not in visited:
                    backtrack(_r, _c, cur_amount, visited)
                    is_end = False
            if is_end:
                self.ans = max(self.ans, cur_amount)
            # Trick: Back Tracking
            #   Backtrack is to recover the state so it don't have any impact on
            #   OUTTER step, but avaialbe for all SUB step;
            #   In this way, we can TRY different choices
            #   Back tracking is used when you need to work on ALL/SOME choices.
            visited.remove((r, c))
                
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    backtrack(r, c)
                    
        return self.ans
```

### [526. Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/)

```python
# Fav
class Solution:
    def countArrangement(self, n: int) -> int:
        def backtrack(perm, start=0):
            # Trick: Permutation with Back Tracking
            if start == n:
                # Starting from 0, collect permutation at n
                return 1
            ans = 0
            
            for j in range(start, n):
                # Trick: Just check the num moved to current start, rather than
                # the cur value at start with j. This is because, you can not skip
                # the next level start + 1 scan just because the first case is
                # not matching. You should check the first part util start matching
                # the remaining items will be checked later using same patter, which is
                # only checking left part until start.
                if perm[j] % (start + 1) == 0 or (start + 1) % perm[j] == 0:
                    perm[start], perm[j] = perm[j], perm[start]
                    ans += backtrack(perm, start + 1)
                    perm[start], perm[j] = perm[j], perm[start]
            return ans
        
        perm = [i + 1 for i in range(n)]
        return backtrack(perm)
```

### [698. Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        def backtrack(partitions):
            if not nums:
                return True
            x = nums.pop()
            for i, partition in enumerate(partitions):
                if partition + x <= target:
                    partitions[i] += x
                    if backtrack(partitions):
                        return True
                    partitions[i] -= x
                # Because for each number, we start from leftmost
                # avaialbe partition, if one partition is empty,
                # all the right most should be empty, and the current
                # empty partition is the begining of remaining empty partitions.
                # For remaining nums and remaing partitions, if one numer cannot
                # work with one empty buckets, others are same, cannot work.
                if not partition:
                    break
            nums.append(x)
            return False
        
        target, rem = divmod(sum(nums), k)
        if rem:
            return False
        nums.sort()
        if nums[-1] > target:
            return False
        while nums and nums[-1] == target:
            nums.pop()
            k -= 1
        return backtrack([0] * k)
```


### [980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/)

```python
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        # Alg: DFS/BFS is to search a graph, Backtracking is to search on
        #   a tree generated based on graph following some rule, e.g. combinations
        #   of the nodes in the graph. Therefor, bracktracking is more complicated 
        #   than simple DFS/BFS
        R, C = len(grid), len(grid[0])
        # Find the start point
        s_r = s_c = None
        obstacles = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    s_r, s_c = r, c
                elif grid[r][c] == -1:
                    obstacles += 1
        opens = R * C - obstacles
        visited, ans = set(), 0
        def dfs(r, c, path=[]):
            nonlocal ans
            path.append((r, c))
            if grid[r][c] == 2 and len(path) == opens:
                ans += 1
            else:
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    _r, _c = r + dr, c + dc
                    if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] != -1 and (_r, _c) not in path:
                        dfs(_r, _c, path)
            path.remove((r, c))
        
        dfs(s_r, s_c)
        return ans

    def uniquePathsIII(self, grid: List[List[int]]) -> int:
            R, C = len(grid), len(grid[0])
            s_r = s_c = None
            non_obstacles = 0
            for r in range(R):
                for c in range(C):
                    if grid[r][c] == 1:
                        s_r, s_c = r, c
                    if grid[r][c] != -1:
                        non_obstacles += 1
            ans = 0
            # Trick: Use remaining rather than complete path
            def backtrack(r, c, remaining):
                nonlocal ans
                if grid[r][c] == 2 and remaining == 1:
                    ans += 1
                    return
                # Trick: Update grid directly for visited status and roll it back
                temp = grid[r][c]
                grid[r][c] = -2
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    _r, _c = r + dr, c + dc
                    if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] not in (-1, -2):
                        backtrack(_r, _c, remaining - 1)
                grid[r][c] = temp

            backtrack(s_r, s_c, non_obstacles)
            return ans
```

### [491. Increasing Subsequences](https://leetcode.com/problems/increasing-subsequences/)

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        # Wrong
        N = len(nums)
        dp = [0] * N
        dp[0] = 0
        for i in range(1, N):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], 2 * dp[j] + 1)
        return dp[-1]
    
    def findSubsequences(self, nums):
        # Time Complexity: O(n ^ 2)
        N, ans = len(nums), set()
        def backtrack(start, seq):
            if len(seq) > 1:
                ans.add(tuple(seq))
            if start == N:
                return
            for i in range(start, N):
                if not seq or seq[-1] <= nums[i]:
                    seq.append(nums[i])
                    backtrack(i + 1, seq)
                    seq.pop()
        backtrack(0, [])
        return ans
            
        
```

### [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}
        
        ans = []
        
        def backtrack(combination, idx):
            if idx >= len(digits):
                ans.append(combination) 
            else:
                for c in phone[digits[idx]]:
                    backtrack(combination + c, idx + 1)
                    
        if digits:
            backtrack('', 0)
        return ans
```

### [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        def gen(s, count_l, count_r):
            if count_l == n and count_r == n:
                ans.append(s)
                return
            if count_l > count_r:
                if count_l < n:
                    gen(s + '(', count_l + 1, count_r)
                gen(s + ')', count_l, count_r + 1)
            elif count_l == count_r:
                gen(s + '(', count_l + 1, count_r)
                
        gen('', 0, 0)
        return ans
```

### [39. Combination Sum](https://leetcode.com/problems/combination-sum/)

### [46. Permutations](https://leetcode.com/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        output = []
        l = len(nums)
        def backtrack(first = 0):
            if first == l:
                output.append(nums[:])
                
            for i in range(first, l):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[first], nums[i] = nums[i], nums[first]
                
        backtrack()
        return output
```

### [78. Subsets](https://leetcode.com/problems/subsets/)

### [79. Word Search](https://leetcode.com/problems/word-search/)

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        self.rows, self.cols = len(board), len(board[0])
        self.board = board
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.backtrack(r, c, word):
                    return True
        return False
                
                
    def backtrack(self, row, col, suffix):
        if len(suffix) == 0:
            return True
        
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols or self.board[row][col] != suffix[0]:
            return False
        
        self.board[row][col] = '#'
        
        di = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ret = False
        
        for d in di:
            if self.backtrack(row + d[0], col + d[1], suffix[1:]):
                ret = True
                break
        
        self.board[row][col] = suffix[0]
        
        return ret
```

### [93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        ans = []
        
        def backtrack(start=len(s) - 1, curr=[], section=0):
            if section == 4 :
                ans.append('.'.join(curr[::-1]))
            
            digits = [0, 1, 2] if section < 3 else [start]
            for d in digits:
                n = s[start - d :start + 1]
                if n == '0' or (n and n[0] != '0' and int(n) < 256):
                    curr.append(n)
                    backtrack(start - d - 1, curr, section + 1)
                    curr.pop()
            
        backtrack()
        return ans
```

### [131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        
        def is_palin(i, j):
            # Trick: Memory - Repeated Subproblem
            if is_palin_memo[i][j] != -1:
                return is_palin_memo[i][j]
            
            # Trick: No need to check '=', even if both i, j +/-1
            #        Because you only need to check util (a)b(c) or (a)(b)
            while i < j:
                if s[i] != s[j]: 
                    is_palin_memo[i][j] = False
                    break
                i, j = i + 1, j - 1
            
            if is_palin_memo[i][j] == -1:
                is_palin_memo[i][j] = True
            
            return is_palin_memo[i][j]
        
        def is_palin2(i, j):
            # Trick: Use python string slice to check palindom
            return s[i:j + 1] == (s[j::-1] if i == 0 else s[j:i - 1:-1])
        

        # Trick: Backtracking
        def backtrack(i=0, result=[]):
            # First hanlde end condition
            if i == len(s):
                self.ans.append(result[:])
            
            for j in range(i, len(s)):
                if is_palin(i, j):
                    result.append(s[i: j + 1])
                    backtrack(j + 1, result)
                    result.pop()
        self.ans = []
        is_palin_memo = [[-1] * len(s) for _ in range(len(s))]
        backtrack()
        return self.ans

    def partition(self, s: str) -> List[List[str]]:
        N, ans = len(s), []
        def is_parlin(s):
            return s == s[::-1]
        def backtrack(i, partitions):
            if i == N:
                # Trick: Copy full array
                ans.append(partitions[:])
                return
            for j in range(i + 1, N + 1):
                cur = s[i:j]
                # print(cur)
                if is_parlin(cur):
                    partitions.append(cur)
                    backtrack(j, partitions)
                    partitions.pop()
        backtrack(0, [])
        return ans
```