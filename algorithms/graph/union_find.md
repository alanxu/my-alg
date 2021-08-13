# Overview
Also called Disjoint Set
### [547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/)

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        N = len(isConnected)
        uf = [x for x in range(N)]
        def find(x):
            if uf[x] == x:
                return x
            # Trick: This is 50% faster than just return find(uf[x])
            parent = find(uf[x])
            uf[x] = parent
            return uf[x]
        
        def union(x, y):
            if find(x) != find(y):
                uf[find(x)] = find(y)
                
        for i in range(N):
            # Trick: Just calc the parts afterwards
            for j in range(i + 1, N):
                if isConnected[i][j] == 1:
                    union(i, j)
        
        # Trick: Still need to check the parent
        return sum([1 for i, x in enumerate(uf) if i == x])

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        N = len(isConnected)
        
        parents = [i for i in range(N)]
        ranks = [0] * N
        
        def find(i):
            if parents[i] != i:
                parents[i] = find(parents[i])
            return parents[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return
            if ranks[rooti] < ranks[rootj]:
                rooti, rootj = rootj, rooti
            parents[rootj] = rooti
            ranks[rooti] += ranks[rootj]
        
        for i, conn in enumerate(isConnected):
            for j, connected in enumerate(conn):
                if connected:
                    union(i, j)
                    
        return len(set([find(i) for i in range(N)]))
```

### [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)

```python
class DSU:
    def __init__(self):
        self.par = [i for i in range(1001)]
        self.rnk = [0] * 1001
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        elif self.rnk[xr] < self.rnk[yr]:
            self.par[xr] = yr
        elif self.rnk[xr] > self.rnk[yr]:
            self.par[yr] = xr
        else:
            self.par[xr] = yr
            self.rnk[yr] += 1
        return True
    
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        dsu = DSU()
        for edge in edges:
            if not dsu.union(*edge):
                return edge
```

### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

```python
class UnionFind:
    def __init__(self, grid: List[List[str]]):
        self.R, self.C = len(grid), len(grid[0])
        self.uf = [-1] * (self.R * self.C)
        self.rank = [0] * (self.R * self.C)
        self.counts = 0
        for r in range(self.R):
            for c in range(self.C):
                if grid[r][c] == '1':
                    self.uf[r * self.C + c] = r * self.C + c
                    self.counts += 1

    def find(self, i):
        if self.uf[i] != i:
            self.uf[i] = self.find(self.uf[i])
        return self.uf[i]
    
    def union(self, i, j):
        rooti, rootj = self.find(i), self.find(j)
        if rooti != rootj:
            if self.rank[rooti] > self.rank[rootj]:
                self.uf[rootj] = rooti
            elif self.rank[rooti] < self.rank[rootj]:
                self.uf[rooti] = rootj
            else:
                self.uf[rootj] = rooti
                self.rank[rooti] += 1
            self.counts -= 1

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        uf = UnionFind(grid)
        # Trick: Can just check right and down
        dirs = [(1, 0), (0, 1)]
        R, C = len(grid), len(grid[0])
        
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    grid[r][c] == '0'
                    for d in dirs:
                        _r, _c = r + d[0], c + d[1]
                        if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] == '1':
                            uf.union(r * C + c, _r * C + _c)

        return uf.counts

    def numIslands(self, grid: List[List[str]]) -> int:
        # Inuition: Use DFS to union
        R, C = len(grid), len(grid[0])
        N = R * C
        parents, rank, counts = [-1] * N, [1] * N, 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    counts += 1
                    parents[r * C + c] = r * C + c
        
        def find(i):
            if parents[i] != i:
                parents[i] = find(parents[i])
            return parents[i]
        def union(i, j):
            nonlocal counts
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return
            if rank[rooti] < rank[rootj]:
                rooti, rootj = rootj, rooti
            parents[rootj] = rooti
            rank[rooti] += rank[rootj]
            counts -= 1
        def dfs(r, c):
            if grid[r][c] == '0':
                return
            i = r * C + c
            grid[r][c] = '0'
            
            for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                j = _r * C + _c
                if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] != '0':
                    union(i, j)
                    dfs(_r, _c)
                    
        for r in range(R):
            for c in range(C):
                dfs(r, c)

        return counts

        def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        self.counts = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    self.counts += 1
        
        parents, rank = [x for x in range(R * C)], [1] * (R * C)
        def find(i):
            if parents[i] != i:
                parents[i] = find(parents[i])
            return parents[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti != rootj:
                if rank[i] < rank[j]:
                    rooti, rootj = rootj, rooti
                parents[rootj] = rooti
                rank[rooti] += rank[rootj]
                self.counts -= 1
        
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    grid[r][c] = '0'
                    # Trick: Only vsite right and down
                    # Trick: Index cell in grid: r * C + c
                    for _r, _c in ((r + 1, c), (r, c + 1)):
                        if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] == '1':
                            union(r * C + c, _r * C + _c)
                            
        return self.counts
    
    def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        def dfs(r, c):
            grid[r][c] = '0'
            for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= _r < R and 0 <= _c < C and grid[_r][_c] == '1':
                    dfs(_r, _c)
        ans = 0
        for r in range(R):
            for c in range(C):
                # Only run dfs() when find a new '1', this prunes the repeated
                # checks
                if grid[r][c] == '1':
                    ans += 1
                    dfs(r, c)

        return ans
```

### [305. Number of Islands II](https://leetcode.com/problems/number-of-islands-ii/)

```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        # Alg - Disjoint Set
        # With path compression and union by rank, the find() and union() method
        # can be amortized O(1)
        # So the total time is O(m * n + L), m * n is time for consturction initial
        # disjoint set, L is num of operations we need to process one by one.
        parents, rank, self.counts = [x for x in range(m * n)], [1] * (m * n), 0
        def find(i):
            # Alg: Disjoint set - path compression
            if parents[i] != i:
                parents[i] = find(parents[i])
            return parents[i]
        def union(i, j):o
            rooti, rootj = find(i), find(j)
            if rooti != rootj:
                # Alg: Disjoint set - union by rank
                if rank[rooti] < rank[rootj]:
                    rooti, rootj = rootj, rooti
                parents[rootj] = rooti
                rank[rooti] += rank[rootj]
                self.counts -= 1
        def add(i):
            self.counts += 1
        
        ans, seen = [], set()
        for p in map(tuple, positions):
            if p in seen:
                # Use seen to handle duplicated operation
                ans.append(self.counts)
                continue
            seen.add(p)
            i = p[0] * n + p[1]
            add(i)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = p[0] + d[0], p[1] + d[1]
                # Use seen to find if cur island can be merged
                if (r, c) in seen:
                    union(i, r * n + c)
            ans.append(self.counts)
        return ans
```

### [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)

```python
class UnionFind:
    def __init__(self, n):
        # Trick: Number of nodes is not given, but we
        # can get a rang of it from len of edges.
        # the max num of nodes is len(edges) + 1, in this
        # case it is a line [(1, 2), (2, 3), (3, 4)...]
        # we don't care global status of union, we just
        # care the state of two nodes at a time, so there
        # are some ununioned nodes doesn't matter
        self.parent = [i for i in range(n + 2)]
        self.rank = [0] * (n + 2)
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        rooti, rootj = self.find(i), self.find(j)
        if rooti == rootj:
            return False
        if self.rank[rooti] < self.rank[rootj]:
            rooti, rootj = rootj, rooti
        self.parent[rootj] = rooti
        self.rank[rooti] += self.rank[rootj]
        return True

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # Intuition: Connect nodes based on the edges one by one by union
        # nodes together, if two nodes already connected (union returns False)
        # the edge needs to be removed.
        # Note the undirected graph only has 1 circle
        uf = UnionFind(len(edges))
        for u, v in edges:
            if not uf.union(u, v):
                return (u, v)
```

### [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [1] * n
        
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        rooti, rootj = self.find(i), self.find(j)
        if self.rank[rooti] < self.rank[rootj]:
            rooti, rootj = rootj, rooti
        self.parent[rootj] = rooti
        self.rank[rooti] += self.rank[rootj]

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # Intuition: Union find on each accounts as input, check if the emails
        # has subset (this is O(n) and is the reason it is 5%).
        uf = UnionFind(len(accounts))
        emails = [set(acc[1:]) for acc in accounts]
        
        n = len(accounts)
        for i in range(n):
            for j in range(i + 1, n):
                if emails[i] & emails[j]:
                    uf.union(i, j)
        
        ans = {}
        for i, acc in enumerate(accounts):
            j = uf.find(i)
            emails = ans.get(j, set())
            emails |= set(acc[1:])
            ans[j] = emails
            
        return [[accounts[i][0]] + sorted(list(emails)) for i, emails in ans.items()]
    
    def accountsMerge(self, accounts):
        # Intuition: Uniion find on each emails, union emails if they are same input account,
        # for diff input accounts but same account, there is >= 1 email got same parnent (find(email)),
        # so that email link all emails in same input account to the other account.
        
        # In the initial inputs, emails are grouped by small groups, but small group can
        # be further grouped to bigger groups. The problem is to get the bigger group.
        # In the whole process, it is mostly emails' game, accout name is just for display.
        
        # There is implicit condition that unique email in diff accounts, the account names are same
        
        # Use email as node in graph rather than accounts. Using accounts will work, we can assume
        # each account is a node, and 2 nodes are connected when there are same email in them. But
        # it takes O(n^2).
        
        dsu = UnionFind(10001)
        em_to_name = {}
        em_to_id = {}
        i = 0
        for acc in accounts:
            name = acc[0]
            for email in acc[1:]:
                em_to_name[email] = name
                if email not in em_to_id:
                    em_to_id[email] = i
                    i += 1
                # Trick: Just need to union first email and every others in account
                dsu.union(em_to_id[acc[1]], em_to_id[email])
        
        # Now we build the disjoint set and know all groups the email can be divide
        # we just have to phisically group the email as a base for final answer.
        # Use a dict, key is the parent of that eamail group, value is a list of
        # all emails in that group
        ans = collections.defaultdict(list)
        for email in em_to_name:
            ans[dsu.find(em_to_id[email])].append(email)

        # Given the email groups dict, output its vlaues (list of emais) after sorting
        # and add the account name in front of each list.
        return [[em_to_name[v[0]]] + sorted(v) for v in ans.values()]
```

### [924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)

```python
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
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
            
        for i in range(N):
            for j in range(i + 1, N):
                if graph[i][j]:
                    union(i, j)
        
        # Intuition: For each node in initial, if there are ones is the unique
        # node in the list in one connected component (连接分量), we return the
        # one with biggest connected component. If there is no unique node in
        # one connected component, return the smallest initial. If there are 
        # multiple nodes in initial in different connected component of same size,
        # return smallest one.
        area = Counter([find(i) for i in range(N)])
        initial_area = Counter(find(i) for i in initial)
        max_area, ans = 0, min(initial)
        for i in initial:
            if initial_area[find(i)] == 1:
                if area[find(i)] > max_area:
                    max_area, ans = area[find(i)], i
                elif area[find(i)] == max_area:
                    ans = min(ans, i)
                    
        return ans
```

### [785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)

### [947. Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/)

```python
# Fav
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        # Brute Force: Backtrack
        # TLE
        map_r, map_c = defaultdict(set), defaultdict(set)
        
        for r, c in stones:
            map_r[r].add((r, c))
            map_c[c].add((r, c))

        def backtrack(removed, map_r, map_c):
            ans = removed
            for r, _stones in map_r.items():
                if len(_stones) > 1:
                    for _stone in list(_stones):
                        map_r[r].remove(_stone)
                        map_c[_stone[1]].remove(_stone)
                        ans = max(ans, backtrack(removed + 1, map_r, map_c))
                        map_r[r].add(_stone)
                        map_c[_stone[1]].add(_stone)
            for c, _stones in map_c.items():
                _stones = list(_stones)
                if len(_stones) > 1:
                    for _stone in _stones:
                        map_c[c].remove(_stone)
                        map_r[_stone[0]].remove(_stone)
                        ans = max(ans, backtrack(removed + 1, map_r, map_c))
                        map_c[c].add(_stone)
                        map_r[_stone[0]].add(_stone)
            return ans
        
        return backtrack(0, map_r, map_c)
    
    def removeStones(self, stones):
        graph = collections.defaultdict(list)
        for i, x in enumerate(stones):
            for j in range(i):
                y = stones[j]
                if x[0]==y[0] or x[1]==y[1]:
                    graph[i].append(j)
                    graph[j].append(i)

        N = len(stones)
        ans = 0

        seen = [False] * N
        for i in range(N):
            if not seen[i]:
                stack = [i]
                seen[i] = True
                while stack:
                    ans += 1
                    node = stack.pop()
                    for nei in graph[node]:
                        if not seen[nei]:
                            stack.append(nei)
                            seen[nei] = True
                ans -= 1
        return ans
    
    def removeStones(self, stones):
        # Intuition: Stones with same row and col are treated as same
        # connected component, if we follow topological order to remove
        # a connected component, there will be just 1 stone left for
        # each connected component. So the problem becomes find the
        # number of connected component k, and answer is len(stones) - k
        
        # Time Complexity: O(N \log N)O(NlogN), where NN is the length of stones. 
        # (If we used union-by-rank, this can be O(N * \alpha(N))O(N∗α(N)), 
        # where \alphaα is the Inverse-Ackermann function.)

        # Space Complexity: O(N)O(N).

        N = 20000
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
        
        for x, y in stones:
            # Trick: Instead of union each stone, we union the x, y of each
            # stone.
            union(x, y + 10000)
        # Trick: We can just check find(x), because find(x) == find(y)
        return len(stones) - len({find(x) for x, y in stones})
```

### [959. Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)

```python

# \0/
# 3X1
# /2\
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        # Intuition: Devid a cell into 4 diagnally. Based on
        # the value of each original cell, we can union 4 smaller
        # cells into 1 (when ' ') or 2 groups. Then union one cell
        # with other cells at N/S/W/E because we know adjacent smaller
        # cells are guaranteed to be 1 group. The number of groups is
        # the ans.
        N = len(grid)
        NN = 4 * N * N
        parent, rank = [i for i in range(NN)], [1] * NN
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
        
        # Trick: A way to iterate a grid
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                root = 4 * (r * N + c)
                if val in '/ ':
                    union(root + 0, root + 1)
                    union(root + 2, root + 3)
                if val in '\\ ':
                    union(root + 0, root + 2)
                    union(root + 1, root + 3)
                
                if r + 1 <  N: union(root + 3, root + 4 * N + 0)
                if r - 1 >= 0: union(root + 0, root - 4 * N + 3)
                if c + 1 <  N: union(root + 2, root + 4 + 1)
                if c - 1 >= 0: union(root + 1, root - 4 + 2)
                    
        return len({find(x) for x in range(NN)})
```

### [990. Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations/)

```python
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        # Intuition: Union Find. Firstly union all variables with equals,
        # then check unquality of variables is not unioned via find().
        # Trick: Generate dict for alphabetic
        # Trick: default dict use a specic value as default
        parent, rank = {c: c for c in string.ascii_lowercase}, defaultdict(lambda: 1)

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return
            if rank[rooti] < rank[rootj]:
                rooti, rootj = rootj, rooti
            parent[rootj] = rooti
            rank[rooti] += rank[rootj]
        
        q = deque()
        # Trick: split a str to mult char variable
        for eq in equations:
            a, op, _, b = eq
            if op == '=':
                union(a, b)
            else:
                q.appendleft(eq)
        
        for a, op, _, b in q:
            if find(a) == find(b):
                return False
            
        return True
```


### [1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/)

```python
class Solution:
    def minimumCost(self, N: int, connections: List[List[int]]) -> int:
        """
        Alg: Kruskal's algo. Find min cost of minimum spanning subtree.
             Initial the disjoint set with each city. Order connections
             by cost. Use lower cost connections to union the city, check
             if the cities for a connection is already connected, if so,
             ignore that connection, else use it.
        """
        parents, rank = [i for i in range(N + 1)], [0] * (N + 1)
        def find(i):
            if parents[i] != i:
                parents[i] = find(parents[i])
            return parents[i]
        def union(i, j):
            rooti, rootj = find(i), find(j)
            if rooti == rootj:
                return False
            if rank[rooti] < rank[rootj]:
                rooti, rootj = rootj, rooti
            parents[rootj] = rooti
            rank[rooti] += rank[rootj]
            return True
        
        connections.sort(key=lambda x: x[2])
        
        ans = 0
        for a, b, cost in connections:
            if union(a, b):
                ans += cost
        
        root = find(N)
        return ans if all(root == find(i) for i in range(1, N + 1)) else -1
```
