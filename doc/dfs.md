## MinMax

### [1563. Stone Game V](https://leetcode.com/problems/stone-game-v/)
https://youtu.be/2_JkASlmxTA
```python
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        N = len(stoneValue)
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N - 1):
            dp[i][i + 1] = max(stoneValue[i], stoneValue[i + 1])
        
        for l in range(3, N):
            for i in range(0, N - l + 1):
                print(i)
                j = i + l - 1
                for k in range(i, j):
                    left_sum, right_sum = sum(stoneValue[i:k + 1]), sum(stoneValue[k + 1:j + 1])
                    if left_sum > right_sum:
                        dp[i][j] = max(dp[i][j], right_sum + dp[k + 1][j])
                    elif left_sum < right_sum:
                        dp[i][j] = max(dp[i][j], left_sum + dp[i][k])
                    else:
                        dp[i][j] = max(dp[i][j], left_sum + max(dp[k + 1][j], dp[i][k]))
        print(dp)
        return dp[0][-1]
    
    def stoneGameV(self, stoneValue: List[int]) -> int:
        n = len(stoneValue)
        acc = [0] + list(itertools.accumulate(stoneValue))

        @functools.lru_cache(None)
        def dfs(i, j):
            if j == i:
                return 0
            ans = 0
            for k in range(i, j):
                s1, s2 = acc[k + 1] - acc[i], acc[j + 1] - acc[k + 1]
                if s1 <= s2:
                    ans = max(ans, dfs(i, k) + s1)
                if s1 >= s2:
                    ans = max(ans, dfs(k + 1, j) + s2)
            return ans

        return dfs(0, n - 1)
    

        def stoneGameV(self, stoneValue: List[int]) -> int:
            n = len(stoneValue)
            acc = [0] + list(itertools.accumulate(stoneValue))

            @functools.lru_cache(None)
            def dfs(i, j):
                if j - i == 1:
                    return 0
                ans = 0
                for k in range(i + 1, j):
                    s1, s2 = acc[k] - acc[i], acc[j] - acc[k]
                    if s1 <= s2:
                        ans = max(ans, dfs(i, k) + s1)
                    if s1 >= s2:
                        ans = max(ans, dfs(k, j) + s2)
                return ans

            return dfs(0, n)
```


## Others

### [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        
        def dfs(r, c):
            if 0 <= r < rows and 0 <= c < cols:
                if board[r][c] == 'O':
                    board[r][c] = 'E'
                    for d in directs:
                        dfs(r + d[0], c + d[1])
        
        
        rows, cols = len(board), len(board[0])
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Trick: Use itertools.product to generate boarder coordinators
        from itertools import product
        boarders = list(product(range(rows), [0, cols - 1])) + list(product([0, rows - 1], range(cols)))
        
        # Trick: It is important to choose what to search, in this case all the boarders and connected 'O's
        for r, c in boarders:
            dfs(r, c)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'E':
                    board[r][c] = 'O'
        
        return board
```

### [785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # Trick: Coloring. Used when you need to divid nodes into groups
        colors = {}
        
        def bfs(root=0):
            q = deque([root])
            while q:
                node = q.pop()
                if node not in colors:
                    colors[node] = 0
                for nei in graph[node]:
                    if nei in colors and colors[nei] == colors[node]:
                        return False
                    if nei not in colors:
                        q.appendleft(nei)
                        colors[nei] = colors[node] ^ 1
            return True
        
        # Because there could be disconnected graphs, so we have to traverse all
        # nodes, and use memory.
        for node in range(len(graph)):
            if not bfs(node):
                return False

        return True
```

```python
class Solution:
    # Trick: Flood Fill
    def shortestBridge(self, A: List[List[int]]) -> int:
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        rows, cols = len(A), len(A[0])
        
        def find_and_mark_one_island():
            r, c = 0, 0
            
            # Find first point in one island as start point
            for i in range(rows):
                for j in range(cols):
                    if A[i][j] == 1:
                        r, c = i, j
                        break
            
            # Traverse first island, mark it as 2 and collect the boarder in a queue
            q = deque()
            
            def dfs(r, c):                

                A[r][c] = 2
                
                for d in directs:
                    _r, _c = r + d[0], c + d[1]
                    if 0 <= _r < rows and 0 <= _c < cols:
                        if A[_r][_c] == 0:
                            if (r, c) not in q: 
                                q.appendleft((r, c))
                        elif A[_r][_c] != 2:
                            dfs(_r, _c)
            dfs(r, c)
            return q
            
        def expand_island_1(boarders):
            # Use multi-source BFS to expand the boarder 1 layer a time util reached to 1 which is island 2
            q = boarders
            
            step = 0
            
            while q:
                Len = len(q)
                for _ in range(Len):
                    r, c = q.pop()
                    for d in directs:
                        _r, _c = r + d[0], c + d[1]
                        if 0 <= _r < rows and 0 <= _c < cols:
                            if A[_r][_c] == 0:
                                q.appendleft((_r, _c))
                                A[_r][_c] = 2
                            elif A[_r][_c] == 1:
                                return step
                                
                step += 1

        q = find_and_mark_one_island() 
        step = expand_island_1(q)
        
        return step
```

### [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

```python
class Solution:
    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        # Find adjcent path from ocean border to node inside, all nodes in path can flow to that ocean
        # To calculate for 2 oceans, do it separatedly, then combine the results.
        # This is a graph traverse problem, searching acending pathes.
        if not matrix: return []
        
        rows, cols = len(matrix), len(matrix[0])
        m = matrix
        flow = [[0] * cols for _ in range(rows)]
        ans = set()
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        PACIFIC, ATLANTIC = 1, 2
        
        # Trick: In DFS recursion, you can determin if current node should be process or process the current
        #        node and determin if nei nodes should be processed. For first approach, you need to pass more
        #        info into recursion func. And make sure visited is marked only when the current node is 
        #        processed.
        # Trick: In DFS recursion func, care about:
        #        1) Should process current node?
        #        2) Process current node
        #        3) Should process nei node?
        #        4) When add to visited
        #        In each call, must process current node, process and mark visit must happen together;
        #        Determin self or neighbor must not happen together. Determine neibghbor, then you can 
        #        process cur node directly in a call; if determin self, just call dfs(nei) directly.
        # Trick: Use bitmask to mark multiple ocean and visited
        #        1 - Pacific, 2 - Atlantic.   flow[r][c] |= ocean - Mark for ocean
        #        flow[_r][_c] & ocean == 0 - Check for ocean
        def dfs(r, c, ocean):
            flow[r][c] |= ocean
            
            for d in directs:
                _r, _c = r + d[0], c + d[1]
                if 0 <= _r < rows and 0 <= _c < cols and flow[_r][_c] & ocean == 0 and m[r][c] <= m[_r][_c]:
                    dfs(_r, _c, ocean)
            
            if flow[r][c] == 3:
                ans.add((r, c))

        from itertools import product
        p_borders = list(product([0], range(cols))) + list(product(range(rows), [0]))
        a_borders = list(product([rows - 1], range(cols))) + list(product(range(rows), [cols - 1]))
        for r, c in p_borders:
            dfs(r, c, PACIFIC)
        for r, c in a_borders:
            dfs(r, c, ATLANTIC)

        return ans
```

### [323. Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        
        # Trick: Coloring.
        # Trick: DFS/BFS will make sure visit all connected nodes in one recursion call
        def dfs(node, color):
            colors[node] = color
            if node in graph:
                for nei in graph[node]:
                    if nei not in colors:
                        dfs(nei, color)
        
        graph = defaultdict(list)
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])
        colors = {}
        color = 0
        for i in range(n):
            if i not in colors:
                dfs(i, color)
                color += 1
                
        return color
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
```

### [430. Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/)

```python
class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        def dfs(node):
            while node:
                if node.child:
                    tail = dfs(node.child)
                    tail.next = node.next
                    if node.next:
                        node.next.prev = tail
                    node.next = node.child
                    node.child.prev = node
                    node.child = None

                if node.next:
                    node = node.next
                else:
                    return node
        dfs(head)
        return head
```

### [694. Number of Distinct Islands](https://leetcode.com/problems/number-of-distinct-islands/)

```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        # Trick: One dfs can find all adjcent nodes, so if a node is seen
        #        it means all its adjcent nodes are seen
        # Trick: Use same dfs func, it make sure the order of each node in
        #        'same' islands is same
        
        rows, cols = len(grid), len(grid[0])
        seen = set()
        shapes = set()
        
        def scan(r, c, r0, c0):
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] and (r, c) not in seen:
                # Trick: Normalize each node in shap based on (r0, c0)
                shape.append((r - r0, c - c0))
                seen.add((r, c))
                for nei in [(r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)]:
                    scan(*nei, r0, c0)
                    
        for r in range(rows):
            for c in range(cols):
                # Define a shape for each starting node, if the starting node is within
                # previous shape, cur shape will be empty
                shape = []
                # For each node in grid, begin scan with the cur node as starting
                # if cur node is in previous shape, it will skip
                scan(r, c, r, c)
                
                if shape:
                    # Trick: Use frozenset
                    shapes.add(frozenset(shape))
        return len(shapes)
```

### [364. Nested List Weight Sum II](https://leetcode.com/problems/nested-list-weight-sum-ii/)

```python
class Solution:
    def depthSumInverse(self, nestedList):
        # https://leetcode.com/problems/nested-list-weight-sum-ii/discuss/83643/Python-solution-with-detailed-explanation
        total_sum, level_sum = 0, 0
        while len(nestedList):
            next_level_list = []
            for x in nestedList:
                if x.isInteger():
                    level_sum += x.getInteger()
                else:
                    for y in x.getList():
                        next_level_list.append(y)
            total_sum += level_sum
            nestedList = next_level_list
        return total_sum
```

### [529. Minesweeper](https://leetcode.com/problems/minesweeper/)
```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        # Trick: DFS for Graph
        #        1) In each recursive call, cur node must have enough info to process it self,
        #        it cannot depends on return of sub reursive calls. There is different from
        #        DFS for tree, because tree doesn't have circular path but graph does.
        #        2) In each recursive call, current node must be processed;
        #        3) A visited set (or equivalent) must be used;
        #        4) Current node must be processed and marked as visited before its childre can be
        #        processed;
        #        5) Conditional check can be put on either cur or childre, depends on condition
        #        
        directs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        rows, cols = len(board), len(board[0])

        def dfs(r, c):
            if board[r][c] == 'M':
                board[r][c] = 'X'
                return
            else:
                m = 0
                for d in directs:
                    _r, _c = r + d[0], c + d[1]
                    if 0 <= _r < rows and 0 <= _c < cols:
                        if board[_r][_c] == 'M': m += 1
                    board[r][c] = str(m) if m else 'B'
            
            if board[r][c] == 'B':
                for d in directs:
                    _r, _c = r + d[0], c + d[1]
                    # board[_r][_c] == 'E' to check visited
                    if 0 <= _r < rows and 0 <= _c < cols and board[_r][_c] == 'E':
                            dfs(_r, _c)
        
        def bfs(r, c):
            q = deque([(r, c)])
            while q:
                r, c = q.pop()
                if board[r][c] == 'M':
                    board[r][c] = 'X'
                    return
                else:
                    m = 0
                    for d in directs:
                        _r, _c = r + d[0], c + d[1]
                        if 0 <= _r < rows and 0 <= _c < cols:
                            if board[_r][_c] == 'M': m += 1
                        board[r][c] = str(m) if m else 'B'
                    
                    if board[r][c] == 'B':
                        for d in directs:
                            _r, _c = r + d[0], c + d[1]
                            # board[_r][_c] == 'E' to check visited
                            if 0 <= _r < rows and 0 <= _c < cols and board[_r][_c] == 'E':
                                    q.appendleft((_r, _c))
                                    # Trick: Mark as pending, so it wont be added again!!
                                    board[_r][_c] = 'P' 
        
        bfs(*click)
        return board
```

### [797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # Trick: Backtracking
        #        If you put path.append() in front of end condition check, it will mess up.
        ans = []
        def backtrack(node=0, path=[]):
            
            if node == len(graph) - 1:
                ans.append(path[:] + [node])
                return
            
            path.append(node)
            for nei in graph[node]:
                backtrack(nei, path)
            path.pop()
        
        
        backtrack()
        return ans
    
    def allPathsSourceTarget2(self, graph: List[List[int]]) -> List[List[int]]:
        # Trick: Top down DP with memo - Faster
        target = len(graph) - 1

        # apply the memoization
        @lru_cache(maxsize=None)
        def allPathsToTarget(currNode):
            if currNode == target:
                return [[target]]

            results = []
            for nextNode in graph[currNode]:
                for path in allPathsToTarget(nextNode):
                    results.append([currNode] + path)

            return results

        return allPathsToTarget(0)
```

### [695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        directs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rows, cols = len(grid), len(grid[0])
        
        def dfs(r, c, area=0):
            if grid[r][c] == 1:
                grid[r][c] = -1
                area += 1
                
                for d in directs:
                    _r, _c = r + d[0], c + d[1]
                    # Trick: DFS for graph
                    #        For non-directional graph, try to do condition check 
                    #        before recursive call on children, it can reduce number of calls
                    if 0 <= _r < rows and 0 <= _c < cols and grid[_r][_c] == 1:
                       area = dfs(_r, _c, area)
                        
            return area
        
        ans = 0
        # Trick: Call dfs one by one with visited set
        for r in range(rows):
            for c in range(cols):
                ans = max(ans, dfs(r, c))
                
        return ans
                
            
```


### [490. The Maze](https://leetcode.com/problems/the-maze/)

```python
# Fav
class Solution:
    def hasPath(self, maze, start, destination):
        # Trick: It is important to understand how the gragh is formed. In this case, it is all points
        #        where the ball stops.
        #        Every problem about from one point to multi possible next points probably are graph/tree problems and can use DFS/BFS
        
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        rows, cols = len(maze), len(maze[0])
        
        def bfs(root):
            q = deque([root])

            while q:
                r, c = q.pop()

                if r == destination[0] and c == destination[1]:
                    return True

                for dr, dc in dirs:
                    _r, _c = r + dr, c + dc
                    # The ball can pass 0 or 2.
                    while 0 <= _r < rows and 0 <= _c < cols and maze[_r][_c] != 1:
                        _r += dr
                        _c += dc
                    _r -= dr
                    _c -= dc

                    # The ball now stops for current step, the current position should be start point
                    # to move in next step
                    # If the current position is analyzed before or is original position, don't don that again (inf loop)
                    if maze[_r][_c] == 0:
                        q.appendleft([_r, _c])
                        # Trick: It also work if you set it at the beginning after pop it
                        maze[r][c] = 2

            return False
        
        def dfs(root):
            r, c = root
            if r == destination[0] and c == destination[1]:
                return True
            maze[r][c] = 2
            for dr, dc in dirs:
                # Trick: Handle Rolling ball
                _r, _c = r + dr, c + dc
                while 0 <= _r < rows and 0 <= _c < cols and maze[_r][_c] != 1:
                    _r += dr
                    _c += dc
                _r -= dr
                _c -= dc
                if maze[_r][_c] == 0:
                    if dfs((_r, _c)):
                        return True
                    
            return False
                    
        return dfs(start)
```

### [464. Can I Win](https://leetcode.com/problems/can-i-win/)

```python
def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        # Intuition:
        # What does 'force a win' mean? It means for player 1, starting from 
        # first round, when he pick a number, there is no way for player 2 to win,
        # no matter what player 2 choose afterword, unless player 1 gives up by
        # choosing a wrong number.
        # Tree is the permutation of selected numbers sum to Total. We can prune it
        # by the remaining numbers are same for different permutation of selected
        # numbers. The anwser is actually if there is a node at node (height == 1),
        # where all it's leaf nodes fail on player1's round (height is odd). But
        # we are not using this to resolve the problem...
        
        # Alg: DFS is to form a tree with permutations of numbers, prune can be used
        # to make it a combination. While build the tree, it also perform specific 
        # logic based on question.
        
        # Intuition: dfs() build/traverse the tree, and return if there is a child node
        # for give node/state, so the cur step can control its win be selecting it.
        @lru_cache(None)
        def dfs(choices, remainder):
            
            if choices[-1] >= remainder:
                return True
            
            for i in range(len(choices)):
                if not dfs(tuple(choices[:i] + choices[i + 1:]), remainder - choices[i]):
                    return True
            
        summed_choices = (maxChoosableInteger + 1) * maxChoosableInteger / 2

        # If all the choices added up are less than the total, no-one can win
        if summed_choices < desiredTotal:
            return False

        # If the sum matches desiredTotal exactly then you win if there's an odd number of turns
        if summed_choices == desiredTotal:
            return maxChoosableInteger % 2
        
        return dfs(tuple(range(1, maxChoosableInteger + 1)), desiredTotal)
```