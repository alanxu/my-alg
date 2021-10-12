# Graph

SP (Shortest Path) problems https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/

## Eulerian Path

### [332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/)

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # Intuition: Eulerian path. The path can be get by doing
        # a post-order traverse and remove the traversed path after
        # used. Why it works? If a node is a start and end of a cycle,
        # it is also the start of non-cycle to end E. By post-order
        # traverse, if the cycle is traverssed first it goes back to
        # start and continue to going to E, E is first (last in final result)
        # of the traverse path, if the cycle first traverse the non-cycle
        # path, E also is the first.
        graph = defaultdict(list)
        for u, v in sorted(tickets, reverse=True):
            graph[u].append(v)
        
        path = []
        def dfs(node):
            while graph[node]:
                dfs(graph[node].pop())
            path.append(node)
            
        dfs('JFK')
        return path[::-1]
```
## Floyd's Tortoise and Hare

### [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Alg: Floyd's Tortoise and Hare 
        # Limitation: if in format of array, nums in array must in range [1:n]
        # and array length is n + 1
        
        # We know that 0 is starting point, cuz all values
        # in nums > 0, meaning no pos pointing to location 0
        tortoise = hare = nums[0]
        
        # Find inersection point, hare will start from there,
        # note hare runs at nums[nums[hare]]
        # Cannot use while tortoise != hare, cuz it doesn't mean
        # t and h are at same location, it is possible t and h
        # are at diff point that pointing to same point (inside
        # and outside of the 'entrence' point). t and h are at
        # 'intersection' point only if t == h and nums[t] == nums[h]
        while True:
            tortoise, hare = nums[tortoise], nums[nums[hare]]
            if tortoise == hare:
                break
        
        # Tortoise start from begining
        # Run until meet in same speed
        tortoise = nums[0]
        while tortoise != hare:
            tortoise, hare = nums[tortoise], nums[hare]
        
        return hare
```

## Tarjan Algorithm and Cycle Detection

### [1059. All Paths from Source Lead to Destination](https://leetcode.com/problems/all-paths-from-source-lead-to-destination/)

```python
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        # Alg: DFS variant - Coloring
        def dfs(node):
            # If the node has color meaning it is
            # either checked (and valid) or in progress.
            # If it already checked, then it has to be
            # valid otherwise the program has returned;
            # If it is in progress, a loop detected, so
            # return False.
            if states[node]:
                return states[node] == BLACK
            # If node is a leaf, it has to be dest, otherwise
            # return False
            if not graph[node]:
                return node == destination
            # Now node is not leaf neither visited before,
            # traverse all nxt nodes see if valid, if any
            # nxt is not valid, the cur node is not valid.
            states[node] = GRAY
            for nxt in graph[node]:
                if not dfs(nxt):
                    return False
            states[node] = BLACK
            return True
        
        graph, states = defaultdict(list), [None] * n
        GRAY, BLACK = 1, 2
        for s, t in edges:
            graph[s].append(t)
        return dfs(source)
```

### [1192. Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/)

```python
class Solution:
    
    rank = {}
    graph = defaultdict(list)
    conn_dict = {}
    
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:

        self.formGraph(n, connections)
        self.dfs(0, 0)
        
        result = []
        for u, v in self.conn_dict:
            result.append([u, v])
        
        return result
            
    def dfs(self, node: int, discovery_rank: int) -> int:
        
        # That means this node is already visited. We simply return the rank.
        if self.rank[node]:
            return self.rank[node]
        
        # Update the rank of this node.
        self.rank[node] = discovery_rank
        
        # This is the max we have seen till now. So we start with this instead of INT_MAX or something.
        min_rank = discovery_rank
        for neighbor in self.graph[node]:
            
            # Skip the parent.
            if self.rank[neighbor] and self.rank[neighbor] == discovery_rank - 1:
                continue
                
            # Recurse on the neighbor.    
            recursive_rank = self.dfs(neighbor, discovery_rank + 1)
            
            # Step 1, check if this edge needs to be discarded.
            if recursive_rank <= discovery_rank:
                del self.conn_dict[(min(node, neighbor), max(node, neighbor))]
            
            # Step 2, update the minRank if needed.
            min_rank = min(min_rank, recursive_rank)
        
        return min_rank
    
    def formGraph(self, n: int, connections: List[List[int]]):
        
        # Reinitialize for each test case
        self.rank = {}
        self.graph = defaultdict(list)
        self.conn_dict = {}
        
        # Default rank for unvisited nodes is "null"
        for i in range(n):
            self.rank[i] = None
        
        for edge in connections:
            
            # Bidirectional edges.
            u, v = edge[0], edge[1]
            self.graph[u].append(v)
            self.graph[v].append(u)
            
            self.conn_dict[(min(u, v), max(u, v))] = 1
```

## Others


### [261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # Trick: A tree of N nodes must have N - 1 edges
        # Trick: Validate graph is tree
        # - N - 1 edges
        # - N nodes can be seen
        if len(edges) != n - 1: return False
        
        # Build graph
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        seen = {0}
        q = deque([0])
        while q:
            node = q.pop()
            seen.add(node)
            for nei in graph[node]:
                if nei not in seen:
                    q.appendleft(nei)
                    
        return len(seen) == n
```

### [1042. Flower Planting With No Adjacent](https://leetcode.com/problems/flower-planting-with-no-adjacent/)

```python
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        for a, b in paths:
            graph[a - 1].append(b - 1)
            graph[b - 1].append(a - 1)
        
        ans = [0] * n
        flowers = {1, 2, 3, 4}
        
        # Trick: If given all nodes, not need to traverse the graph
        for i in range(0, n):
            used = set()
            for nei in graph[i]:
                # Trick: Use dp to get flower from previous gardons
                used.add(ans[nei])
            # Trick: Select an unused value using subtraction of sets
            flower = (flowers - used).pop()
            ans[i] = flower
        return ans
```

### [1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

```python
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        graph = defaultdict(list)
        for f, t, w in edges:
            graph[f].append((t, w))
            graph[t].append((f, w))
        
        ans, max_reachable = -1, math.inf
        for i in range(n):
            dist, heap = {}, [(0, i)]
            while heap:
                d, node = heapq.heappop(heap)
                # Trick: The check can be placed here, or before push nei
                #   here is faster than the other.
                if node in dist: continue
                dist[node] = d
                for nei, d2 in graph[node]:
                    if d + d2 <= distanceThreshold:
                        heapq.heappush(heap, (d + d2, nei))

            if len(dist) - 1 <= max_reachable:
                max_reachable = len(dist) - 1
                ans = i
            
        return ans
```

### [399. Evaluate Division](https://leetcode.com/problems/evaluate-division/)

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # Trick: Create dynamic 2-demension structure
        graph = defaultdict(defaultdict)
        
        for (dividend, divisor), value in zip(equations, values):
            graph[dividend][divisor] = value
            graph[divisor][dividend] = 1 / value
        
        # Trick: Use Back Track to search a path in a Graph
        def backtrack(cur_node, tgt_node, acc_product=1, cur_path=set()):
            cur_path.add(cur_node)
            ret = -1.0
            if tgt_node in graph[cur_node]:
                ret = acc_product * graph[cur_node][tgt_node]
            else:
                for nei, value in graph[cur_node].items():
                    if nei in cur_path:
                        continue

                    ret = backtrack(nei, tgt_node, acc_product * value, cur_path)
                    if ret != -1.0:
                        break
            cur_path.remove(cur_node)
            return ret
        
        ans = []
        for dividend, divisor in queries:
            if dividend not in graph or divisor not in graph:
                ans.append(-1.0)
            elif dividend == divisor:
                ans.append(1.0)
            else:
                ans.append(backtrack(dividend, divisor))
        return ans
```

### [841. Keys and Rooms](https://leetcode.com/problems/keys-and-rooms/)

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited, q = {0}, deque([0])
        while q:
            room = q.pop()
            for nei in rooms[room]:
                if nei not in visited:
                    q.appendleft(nei)
                    visited.add(nei)
        return len(visited) == len(rooms)
```


### [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/):

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # Trick: Treat each account as node in graph, and the common email indicates
        # the connection of the nodes. The problem is to find
        # all the connected nodes in the graph, and they are same account
        
        # Build graph using an email map, connect the accounts when have common
        # email
        email_user_map = defaultdict(list)
        for i, acc in enumerate(accounts):
            for email in acc[1:]:
                email_user_map[email].append(i)
        
        # Give an account, DFS on the graph, find connected accounts to that accounds, 
        # and collect all the emails of each connected nodes and return it
        def dfs(acc, emails=None):
            # Trick: If you use default value of parameter
            #   the value will be persisted between each call without it
            #   guess the function runs within a function, so the data is
            #   saved within the stack
            if not emails: emails = set()
            seen.add(acc)
            emails |= {*accounts[acc][1:]}
            # print(acc)
            # print(emails)
            
            for email in accounts[acc][1:]:
                for nei in email_user_map[email]:
                    if nei not in seen:
                        emails |= dfs(nei, emails)
            return emails
        
        # Trick: The seen set not only help each dfs, also
        #   used to determin if a node need to be traversed as start
        seen = set()
        ans = []
        for i, acc in enumerate(accounts):
            # print(dfs(i))
            # print(acc)
            if i not in seen:
                row = [acc[0]]
                # print(f"--- {i} ---")
                row.extend(sorted(dfs(i)))
                ans.append(row)
            # print(ans)
            
        return ans
```

