
## Clone Graph

### [133. Clone Graph](https://leetcode.com/problems/clone-graph/)
### [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)


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

    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # Intuition: No need coloring, iterate n nodes, dfs on each one
        # when you meet a un-seen node, ans increase by 1,because the
        # node is disconnected from components marked by prev dfs.
        graph, seen = defaultdict(list), set()
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        
        def dfs(node):
            seen.add(node)
            for nei in graph[node]:
                if nei not in seen:
                    dfs(nei)
        
        ans = 0
        for i in range(n):
            if i not in seen:
                ans += 1
                dfs(i)
        return ans
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
        # Intuition: Use lefttop point as anchor and use the relative distance pair to
        # record a shape.
        
        rows, cols = len(grid), len(grid[0])
        seen = set()
        shapes = set()
        
        def scan(r, c, r0, c0):
            # Trick: Check seen and set seen both at parent level
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] and (r, c) not in seen:
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
                # The 'M' can be reached only when it is initially triggered by click, so it is
                # always the first call rather than a recursive call. So we can simply return
                # and this will terminate the game.
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
                                    board[_r][_c] = 'P'
        
        bfs(*click)
        return board
```

### [797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

```python
# Fav
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # Intuition: Backtracking
        #        If you put path.append() in front of end condition check, it will mess up.
        ans = []
        def backtrack(node=0, path=[]):
            # Because path is top down, so path is always different, so backtrack cannot utilize
            # caching, and pruning not always working, e.g. this case.
            
            if node == len(graph) - 1:
                ans.append(path[:] + [node])
                return
            
            path.append(node)
            for nei in graph[node]:
                backtrack(nei, path)
            path.pop()
        
        backtrack()
        return ans
    
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # Intuition: Top down DP with memo - Faster
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
                
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        ans = 0
        def dfs(r, c):
            self.area += 1
            grid[r][c] = 0
            for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= _r < rows and 0 <= _c < cols and grid[_r][_c] == 1:
                    dfs(_r, _c)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    self.area = 0
                    dfs(r, c)
                    ans = max(ans, self.area)

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

### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

### [827. Making A Large Island](https://leetcode.com/problems/making-a-large-island/)

```python
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        # Intuition: Similar as 200, use DFS to mark each island with unique index.
        # Traverse all 0's, find 0 connect to most island and has max size.
        # Corner case: No 1 or no 0
        rows, cols = len(grid), len(grid[0])
        sizes = defaultdict(int)
        def dfs(r, c, idx):
            grid[r][c] = idx
            sizes[idx] += 1
            for _r, _c in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                if 0 <= _r < rows and 0 <= _c < cols and grid[_r][_c] == 1:
                    dfs(_r, _c, idx)
        
        idx = 2
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    dfs(r, c, idx)
                    idx += 1
        
        if not sizes:
            return 1
        
        max_size = rows * cols
        if sum(sizes.values()) in (max_size, max_size - 1):
            return max_size
        
        ans = max(sizes.values())
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    islands = set()
                    for _r, _c in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                        if 0 <= _r < rows and 0 <= _c < cols:
                            islands.add(grid[_r][_c])
                    size = 0
                    for island in islands:
                        size += sizes[island]
                    ans = max(ans, size + 1)
        return ans
```

### [785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)

```python
# Fav
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        color = {}
        def dfs(node):
            if node not in color:
                color[node] = 0
            for nei in graph[node]:
                if nei in color and color[nei] == color[node]:
                    return False
                if nei not in color:
                    color[nei] = color[node] ^ 1
                    if not dfs(nei):
                        return False
            return True
        
        for i in range(len(graph)):
            # Why this dfs is slower than the other dfs?
            # It does a lot of duplicated check on nodes
            # already checked in prevous run.
            if not dfs(i):
                return False
        return True

    def isBipartite(self, graph: List[List[int]]) -> bool:
        N = len(graph)
        parent, rank = [i for i in range(N)], [1] * N
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return
            if rooti < rootj:
                rooti, rootj = rootj, rooti
            parent[rootj] = rooti
            rank[rooti] += rank[rootj]
        
        for u in range(N):
            for v in graph[u][1:]:
                union(graph[u][0], v)
        
        for u in range(N):
            for v in graph[u]:
                if find(u) == find(v):
                    return False
        return True
    
    def isBipartite(self, graph: List[List[int]]) -> bool:
        N = len(graph)
        parent, rank = [i for i in range(N)], [1] * N
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return
            if rooti < rootj:
                rooti, rootj = rootj, rooti
            parent[rootj] = rooti
            rank[rooti] += rank[rootj]
        
        for u in range(N):
            for v in graph[u]:
                if find(u) == find(v):
                    return False
                union(graph[u][0], v)

        return True
    
    def isBipartite(self, graph):
        color = {}
        def dfs(pos):
            for i in graph[pos]:
                if i in color:
                    if color[i] == color[pos]:
                        return False
                else:
                    color[i] = 1 - color[pos]
                    if not dfs(i):
                        return False
            return True
        for i in range(len(graph)):
            if i not in color:
                color[i] = 0
                if not dfs(i):
                    return False
        return True
    
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # Trick: Coloring. Used when you need to divid nodes into groups
        colors = {}
        
        def bfs(root=0):
            q = deque([root])
            while q:
                node = q.pop()
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
            if node not in colors:
                colors[node] = 0
                if not bfs(node):
                    return False

        return True
```

### [778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/)


### [329. Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # No need to use seen dict, cause it look for bigger number
        rows, cols = len(matrix), len(matrix[0])

        @functools.lru_cache(None)
        def dfs(r, c):
            ans = 1
            for _r, _c in (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1):
                if 0 <= _r < rows and 0 <= _c < cols and matrix[_r][_c] > matrix[r][c]:
                    ans = max(ans, dfs(_r, _c) + 1)
            return ans
        
        ans = 0
        for r in range(rows):
            for c in range(cols):
                ans = max(ans, dfs(r, c))
        
        return ans
```