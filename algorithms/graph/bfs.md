# Breadth First Search

When you are given an array or a string and some rule that can group the
items, you have a tree/graph structure.

Sometimes you can use memorization to improve performance.

BFS for graph need visited set, while BFS for tree doesn't need

Good artical: https://leetcode.com/problems/graph-valid-tree/

## Problems

### [934. Shortest Bridge](https://leetcode.com/problems/shortest-bridge/)
```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        # Alg: Flood Fill
        # Find all border nodes of ONE islend as flood, flood-fill starting from the
        # borders until reach to another island. Different nums need to be used
        # to mark different nodes:
        # - 0 water
        # - 1 target island
        # - 2 flood
        R, C = len(grid), len(grid[0])
        
        # Step 1: Find and flood-fill first island and capture the border of it to
        # flood-fill others
        # ------------------------------------------------------------------------
        
        # Get first land of first island
        r, c = 0, 0
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 1:
                    r, c = i, j
                    break
        # Create a queue to capture borders for step 2.
        q = deque()
        # Use dfs to scan first island and flood-fill it
        def dfs(r, c):
            grid[r][c] = 2
            for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= _r < R and 0 <= _c < C:
                    if grid[_r][_c] == 1:
                        dfs(_r, _c)
                    elif grid[_r][_c] == 0 and (_r, _c) not in q:
                        # Trick: use dfs() to capture borders
                        q.append((r, c))
        dfs(r, c)
        
        # Step 2: Flood-fill from borders of 1st island until reach other island
        # using multi-source BFS
        # ----------------------------------------------------------------------
        steps = 0
        while q:
            L = len(q)
            for _ in range(L):
                r, c = q.pop()
                for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= _r < R and 0 <= _c < C:
                        if grid[_r][_c] == 0:
                            q.appendleft((_r, _c))
                            # Flood-fill the water
                            grid[_r][_c] = 2
                        elif grid[_r][_c] == 1:
                            return steps
            steps += 1
        return 0
```

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

### [127. Word Ladder](https://leetcode.com/problems/word-ladder/)

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        all_combo = collections.defaultdict(list)
        
        l = len(beginWord)
        
        for w in wordList:
            for i in range(l):
                all_combo[w[:i] + '*' + w[i + 1:]].append(w)
                
        q = collections.deque([(beginWord, 1)])
        visited = set()
        visited.add(beginWord)
        
        while q:
            cur_w, level = q.popleft()
            
            for i in range(l):
                int_w = cur_w[:i] + '*' + cur_w[i + 1:]
                    
                for w in all_combo[int_w]:
                    if w == endWord:
                        return level + 1
                    
                    if w not in visited:
                        visited.add(w)
                        q.append((w, level + 1))
                        
                all_combo[int_w] = []
                
        return 0
```

### [909. Snakes and Ladders](https://leetcode.com/problems/snakes-and-ladders/)

```python
from functools import lru_cache
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        
        N = len(board)

        def get(s):
            # Given a square num s, return board coordinates (r, c)
            quot, rem = divmod(s-1, N)
            row = N - 1 - quot
            col = rem if row%2 != N%2 else N - 1 - rem
            return row, col
        
        rows, cols = len(board), len(board[0])
        
        @lru_cache
        def get_location(idx):
            row = rows - (idx - 1) // rows - 1
            col = (idx - 1) % cols if row % 2 != rows % 2 else (cols - (idx - 1) % cols -1)
            return (row, col)
        
        q = deque([1])
        visited = set([1])
        step = 0
        
        while q:
            print(f'-- step {step} --')
            l = len(q)
            for i in range(l):
                s = q.popleft()
                
                for j in range(1, 7):
                    ss = s + j
                    
                    
                        
                    if ss > rows * cols:
                        break
                    
                    row, col = get(ss)
                    
                    print((ss, row, col))
                    
                    ss_v = board[row][col]
                    
                    if ss_v == -1 and ss not in visited:
                        print(f'{s} -> {ss}')
                        if ss == rows * cols:
                            return step + 1
                        q.append(ss)
                        visited.add(ss)
                        
                    elif ss_v != -1 and ss_v not in visited:
                        print(f'{s} -> {ss} -> {ss_v}')
                        if ss_v == rows * cols:
                            return step + 1
                        q.append(ss_v)
                        visited.add(ss_v)
                        
            step += 1
        return -1
```

### [934. Shortest Bridge](https://leetcode.com/problems/shortest-bridge/)

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


### [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

```python
from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        queue = deque()

        # Step 1). build the initial set of rotten oranges
        fresh_oranges = 0
        ROWS, COLS = len(grid), len(grid[0])
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh_oranges += 1

        # Mark the round / level, _i.e_ the ticker of timestamp
        queue.append((-1, -1))

        # Step 2). start the rotting process via BFS
        minutes_elapsed = -1
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while queue:
            row, col = queue.popleft()
            if row == -1:
                # We finish one round of processing
                minutes_elapsed += 1
                if queue:  # to avoid the endless loop
                    queue.append((-1, -1))
            else:
                # this is a rotten orange
                # then it would contaminate its neighbors
                for d in directions:
                    neighbor_row, neighbor_col = row + d[0], col + d[1]
                    if ROWS > neighbor_row >= 0 and COLS > neighbor_col >= 0:
                        if grid[neighbor_row][neighbor_col] == 1:
                            # this orange would be contaminated
                            grid[neighbor_row][neighbor_col] = 2
                            fresh_oranges -= 1
                            # this orange would then contaminate other oranges
                            queue.append((neighbor_row, neighbor_col))

        # return elapsed minutes if no fresh orange left
        return minutes_elapsed if fresh_oranges == 0 else -1
```

### [1197. Minimum Knight Moves](https://leetcode.com/problems/minimum-knight-moves/)

```python
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        directions = [(1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)]
        q = collections.deque([(0, 0)])
        visited = set()
        
        step = 0
        while q:
            length = len(q)
            
            for i in range(length):
                p = q.popleft()
                
                if p in visited:
                    continue
                
                if p[0] == abs(x) and p[1] == abs(y):
                    return step
                
                visited.add(p)
                
                for d in directions:
                    pp = (p[0] + d[0], p[1] + d[1])
                    if pp not in visited and pp[0] >= -1 and pp[1] >= -1:
                        q.append(pp)
                    
                
            step += 1
            
        return 0
```

## Flood Fill

### [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        """
        Intuition: Flood fill and Multi-source BFS
        
        Manhattan distance: the distance between two cells (x0, y0) and (x1, y1) is |x0 - x1| + |y0 - y1|.
        It means distance is vertical + horizental
        """
        rows, cols = len(grid), len(grid[0])
        q = collections.deque()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]:
                    q.append((r, c))
        
        if len(q) == rows * cols or len(q) == 0:
            return -1
        
        level = 0
        while q:
            l = len(q)
            for i in range(l):
                r, c = q.popleft()
                for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= _r < rows and 0 <= _c < cols and grid[_r][_c] == 0:
                        grid[_r][_c] = 1
                        q.append((_r, _c))
            level += 1
        return level - 1
```


## Others

### [301. Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        We are required to return the minimum number of invalid parentheses to remove.

        Let's model the problem as a graph,

        node: all possible string by removing parenthesis (The start node is `s`)
        edge (from u to v): by removing a parentheses of u
        As a result, the problem becomes to get the shortest distance from s to a valid node (assuming at level l) in 
        the first place; then get all valid nodes within level l.

        Shortest-path problem is natural to BFS.
        
        https://youtu.be/NWAseBzZj-c
        """
        q = deque([s])
        seen, ans = set(), []
        
        def is_valid(s):
            count = 0
            for x in s:
                if x == '(':
                    count += 1
                elif x == ')':
                    count -= 1
                if count < 0:
                    return False
            return count == 0
        
        while q:
            l = len(q)
            for _ in range(l):
                s = q.popleft()
                if is_valid(s):
                    ans.append(s)
            
                if not ans:
                    for i in range(len(s)):
                        if s[i] in '()':
                            nei = s[:i] + s[i + 1:]
                            if nei not in seen:
                                q.append(nei)
                                seen.add(nei)
                            
        return ans
```


### [1730. Shortest Path to Get Food](https://leetcode.com/problems/shortest-path-to-get-food/)

```python
class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        r = c = 0
        for _r in range(rows):
            for _c in range(cols):
                if grid[_r][_c] == "*":
                    r, c = _r, _c
                    break
                    
        q = deque([(r, c, 0)])
        grid[r][c] = '-'
        while q:
            r, c, step = q.popleft()
            
            for _r, _c in (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1):
                if 0 <= _r < rows and 0 <= _c < cols:
                    if grid[_r][_c] == 'O':
                        grid[_r][_c] = '-'
                        q.append((_r, _c, step + 1))
                    elif grid[_r][_c] == '#':
                        return step + 1
        return -1
```