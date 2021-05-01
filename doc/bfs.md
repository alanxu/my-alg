# Breadth First Search

When you are given an array or a string and some rule that can group the
items, you have a tree/graph structure.

Sometimes you can use memorization to improve performance.

BFS for graph need visited set, while BFS for tree doesn't need

Good artical: https://leetcode.com/problems/graph-valid-tree/

## Problems

### [139. Word Break](https://leetcode.com/problems/word-break/)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = len(s)
        q = collections.deque()
        q.appendleft(0)
        visited = set()
        
        while len(q) > 0:
            i = q.pop()
            
            for j in range(i, l):
                if j in visited:
                    continue
                
                w = s[i:j+1]
                if w in wordDict:
                    q.appendleft(j+1)
                    visited.add(j)
                    
                    if j == l - 1:
                        return True
        return False
```

### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/)

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        
        # Trick: Multi-source BFS
        #        Use 0s as sources, use value to mark if the node is visited (0, max_int and other value).
        #        Each source's nei is scaned, and sources with lower dist is processed ealier, then next level of sources.
        #        Because of above, when a source's nei is able to updated based on source, it is ganranteed to be correct.
        #        Node updated need to be added to queue for next level process.
        #        Unique values in dist are consecutive.
        #        Dist matrix is for result, also for track dist of sources.
        
        m = matrix
        rows, cols = len(m), len(m[0])
        dist = [[math.inf] * cols for _ in range(rows)]
        q = collections.deque()
        dirc = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        for r in range(rows):
            for c in range(cols):
                if m[r][c] == 0:
                    # Set dist for 0s
                    dist[r][c] = 0
                    # Push 0s as first level source
                    q.appendleft((r, c))
        
        # No need to handle len of each level
        while q:
            r, c = q.pop()
            
            # Trick: Navigate in a matrixs
            for d in dirc:
                _r, _c = r + d[0], c + d[1]
                if 0 <= _r < rows and 0 <= _c < cols:
                    if dist[_r][_c] > dist[r][c] + 1:
                        # Set dist based on pre level source
                        dist[_r][_c] = dist[r][c] + 1
                        # Add current level source
                        q.appendleft((_r, _c))
        
        return dist
```


### [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

```python
# Fav
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
                        
        def bfs(r, c):
            q = deque([(r, c)])
            while q:
                # Trick: One line change from q.pop() -> q.popleft() to make bfs to dfs
                row, col = q.pop()
                if board[row][col] != 'O':
                    continue
                board[row][col] = 'E'
                # Trick: Another way to example 4 nei
                if row > 0: q.appendleft((row - 1, col))
                if col > 0: q.appendleft((row, col - 1))
                if row < rows - 1: q.appendleft((row + 1, col))
                if col < cols - 1: q.appendleft((row, col + 1))
                
        rows, cols = len(board), len(board[0])
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Trick: Use itertools.product to generate boarder coordinators
        from itertools import product
        boarders = list(product(range(rows), [0, cols - 1])) + list(product([0, rows - 1], range(cols)))
        
        # Trick: It is important to choose what to search, in this case all the boarders and connected 'O's
        for r, c in boarders:
            # Either bfs or dfs works
            bfs(r, c)
            # dfs(r, c)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'E':
                    board[r][c] = 'O'
        
        return board
```

### [133. Clone Graph](https://leetcode.com/problems/clone-graph/)

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        visited = {}
        def dfs(node):
            if node in visited:
                return visited[node]
            clone = Node(node.val)
            visited[node] = clone
            for nei in node.neighbors:
                clone.neighbors.append(dfs(nei))
            return clone
        
        def bfs(node):
            q = deque([node])
            visited = {node: Node(node.val)}
            while q:
                nd = q.pop()
                clone = visited[nd]
                for nei in nd.neighbors:
                    if nei not in visited:
                        cln = Node(nei.val)
                        visited[nei] = cln
                        q.appendleft(nei)
                    clone.neighbors.append(visited[nei])
                    
            return visited[node]
                
        if not node:
            return None
        
        # return dfs(node)
        return bfs(node)
```

### [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        for row in grid:
            print(row)
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        N = len(grid) - 1
        def bfs(r, c):
            q = deque([(r, c)])
            step = 1
            while q:
                Len = len(q)
                for i in range(Len):
                    r, c = q.pop()
                    
                    # print((r, c))
                    if r == N and c == N:
                        return step
                    for d in directs:
                        _r, _c = r + d[0], c + d[1]
                        if 0 <= _r <= N and 0 <= _c <= N and not grid[_r][_c]:
                            q.appendleft((_r, _c))
                            # Trick: Set enqueued nei to other value to avoid inf loop.
                            # The node is marked first time it is explored, which
                            # means later visit on this node is same or more steps.
                            # So it wont impact the result, if it prunes some paths.
                            # The first node is not marked, so it is added twice and marked
                            # here.
                            # Instead of use -1, can use distance value, and +1 for nei, in this
                            # way no need the step and layer scan
                            grid[_r][_c] = -1
                step += 1
            return -1
                
        return bfs(0, 0) if not grid[0][0] else -1
```

### [752. Open the Lock](https://leetcode.com/problems/open-the-lock/)

```python
# Fav
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:

        # Trick: Because each digit of the number can increase/decrease, so same num can be produced in different steps/ways,
        #        so this is a graph problem.
        # Trick: For shortest path problem of graph, DFS cannot work. Because, for duplicated nodes in a graph, BFS always guarantee the shortest come first,
        #        but DFS does not.
        def bfs():
            def neighbors(node):
                for i in range(4):
                    x = int(node[i])
                    for d in (-1, 1):
                        y = (x + d) % 10
                        yield node[:i] + str(y) + node[i+1:]
            
            dead = set(deadends)
            q = deque([('0000', 0)])
            visited = {'0000'}
            while q:
                num, step = q.pop()
                
                if num == target: return step
                
                # This can be put before enqueue, it is here just to handle edge case
                # when '0000' is deadends
                if num in dead: continue

                for nei in neighbors(num):
                    if nei not in visited:
                        # Trick: BFS must set vistied right before/after it is added to the queue
                        #        dont set visited when the node is poped/processed, cuz duplicate will be added
                        #        to queue before it is processed
                        visited.add(nei)
                        q.appendleft((nei, step + 1))
            return -1

        return bfs()
```


### [286. Walls and Gates](https://leetcode.com/problems/walls-and-gates/)

```python
class Solution:
    def wallsAndGates1(self, rooms: List[List[int]]) -> None:
        # Trick: DFS for graph
        #        1) Process - either cur or children
        #        2) Check if process - either cur or children
        #        3) Vistied - either in place or separated set
        rows, cols = len(rooms), len(rooms[0])
        def bfs(r, c):
            q = deque([(r, c, 0)])
            
            visited = {(r, c)}
            while q:
                r, c, d = q.pop()
                
                for _r, _c in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                    if 0 <= _r < rows and 0 <= _c < cols and rooms[_r][_c] > 0 and (_r, _c) not in visited:
                        rooms[_r][_c] = min(rooms[_r][_c], d + 1)
                        q.appendleft((_r, _c, d + 1))
                        visited.add((_r, _c))
                        
        for r in range(rows):
            for c in range(cols):
                if rooms[r][c] == 0:
                    bfs(r, c)
                    
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        # Trick: For shortest path problem of multi nodes, you can scan each node with BFS, or you can use multi-source BFS.
        #        First node reached by first source is the shortest across all source, other paths are longer or equal to it.
        #        Another benefit is that you dont need another visited set.
        rows, cols = len(rooms), len(rooms[0])
        q = deque()
        
        for r in range(rows):
            for c in range(cols):
                if rooms[r][c] == 0:
                    q.appendleft((r, c, 0))
                    
        while q:
            r, c, d = q.pop()
            for _r, _c in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                    if 0 <= _r < rows and 0 <= _c < cols and rooms[_r][_c] == 2147483647:
                        rooms[_r][_c] = d + 1
                        q.appendleft((_r, _c, d + 1))
```

### [339. Nested List Weight Sum](https://leetcode.com/problems/nested-list-weight-sum/)

```python
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        self.ans = 0
        def dfs(node=nestedList, depth=1):
            if isinstance(node, list):
                for child in node:
                    dfs(child, depth)
            elif node.isInteger():
                self.ans += node.getInteger() * depth
            else:
                dfs(node.getList(), depth + 1)
                
        def bfs():
            # Trick: Handle the depth/layer/steps in BFS
            #        1) attach depth when enqueue
            #        2) use len(q) or for _ in range(len(q))
            #        3) attache '#' at the end of each layer
            q = deque(list(map(lambda node: (node, 1), nestedList)))
            while q:
                node, depth = q.pop()
                if node.isInteger():
                    self.ans += node.getInteger() * depth
                else:
                    # Trick: extendleft()
                    q.extendleft(list(map(lambda node: (node, depth + 1), node.getList())))
        
        bfs()
        return self.ans
```

### [1654. Minimum Jumps to Reach Home](https://leetcode.com/problems/minimum-jumps-to-reach-home/)

```python
class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        q, seen = deque([(0, True)]), {(0, True)}
        # The key is to know what is the limit of forward,
        # limit is a experimental value...
        steps, limit = 0, 5997
        
        for pos in forbidden:
            seen.add((pos, True)) 
            seen.add((pos, False))
        
        while q:
            for _ in range(len(q)):
                pos, dirt = q.pop()
                
                if pos == x:
                    return steps
                
                forward, backward = (pos + a, True), (pos - b, False)
                
                if forward[0] <= limit and forward not in seen:
                    seen.add(forward)
                    q.appendleft(forward)
                
                if dirt and backward[0] > 0 and backward not in seen:
                    seen.add(backward)
                    q.appendleft(backward)
                    
            steps += 1
        
        return -1
```

### [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

```python
# Fav
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
                        
        def bfs(r, c):
            q = deque([(r, c)])
            while q:
                # Trick: One line change from q.pop() -> q.popleft() to make bfs to dfs
                row, col = q.pop()
                if board[row][col] != 'O':
                    continue
                board[row][col] = 'E'
                # Trick: Another way to example 4 nei
                if row > 0: q.appendleft((row - 1, col))
                if col > 0: q.appendleft((row, col - 1))
                if row < rows - 1: q.appendleft((row + 1, col))
                if col < cols - 1: q.appendleft((row, col + 1))
                
        def bfs1(r, c):
            q = deque([(r, c)])
            while q:
                r, c  = q.pop()
                # Trick: Have to check if eligible, cuz there might be multiple
                # node put in queue, and first one is set to 'E'
                if board[r][c] != 'O':
                    continue
                board[r][c] = 'E'
                for d in directs:
                    _r, _c = r + d[0], c + d[1]
                    if 0 <= _r < rows and 0 <= _c < cols and board[_r][_c] == 'O':
                        q.appendleft((_r, _c))
        
        rows, cols = len(board), len(board[0])
        directs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Trick: Use itertools.product to generate boarder coordinators
        from itertools import product
        boarders = list(product(range(rows), [0, cols - 1])) + list(product([0, rows - 1], range(cols)))
        
        # Trick: It is important to choose what to search, in this case all the boarders and connected 'O's
        for r, c in boarders:
            # Either bfs or dfs works
            bfs(r, c)
            # dfs(r, c)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'E':
                    board[r][c] = 'O'
        
        return board
```

### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

