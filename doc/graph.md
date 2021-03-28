# Graph

SP (Shortest Path) problems https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/


### [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Trick: Dijkstra's Algo
        #   For calculate shorted path between vertaces in 
        #   directional/nodirectional weighted graph;
        #   Nondirectional graph can be changed to directional;
        #   It might or might not work with negative paths.

        # Build graph
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Use a heap in order to get the shorted pending vertace
        # for next step processing
        heap = [(0, k)]
        
        # Define distance dict
        dist = {}
        
        # Do BFS
        while heap:
            # When choose the next node to process,
            # Always pick the shortes vatece, it will guarantee
            # the dist for each of its nei is shortest the first
            # time they are calculated
            d, node = heapq.heappop(heap)
            
            # If node already visited, simple ignore
            # This works because we are using heap to process
            # the shortest child first
            if node in dist: continue
            dist[node] = d
            
            for nei, d2 in graph[node]:
                if nei not in dist:
                    heapq.heappush(heap, (d + d2, nei))
                    
        return max(dist.values()) if len(dist) == n else -1
```

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

### [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

```python
class Solution:
    
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Trick: Topological Sort
        #   For DAG, it is not traverse, it is sort, to find the sequence of execution.
        #   Use 3 colors to mark node status:
        #   - WHITE - not visited
        #   - GRAY - the node itself or its successors are currently being checked, used to detact cyclic
        #   - BLACK - the node has been checked
        from collections import defaultdict
        dg = defaultdict(list)
        
        WHITE = 1
        GRAY = 2
        BLACK = 3
        
        for dest, src in prerequisites:
            dg[src].append(dest)
        
        topological_sorted_order = []
        is_possible = True
        color = {k: WHITE for k in range(numCourses)}
        
        def dfs(node):
            nonlocal is_possible
            
            if not is_possible:
                return
            
            color[node] = GRAY
            
            if node in dg:
                for neighbor in dg[node]:
                    if color[neighbor] == WHITE:
                        dfs(neighbor)
                    elif color[neighbor] == GRAY:
                        is_possible = False
                        
            color[node] = BLACK
            
            topological_sorted_order.append(node)
            
        for vertex in range(numCourses):
            if color[vertex] == WHITE:
                dfs(vertex)
                
        return topological_sorted_order[::-1] if is_possible else []
```


### [444. Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/)

```python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
        # Trick: Topological Sort
        
        # Trick: 2 For
        nodes = {x  for seq in seqs for x in seq}
        
        if len(org) != len(nodes):
            # If nodes in seqs not equal to org, False
            return False
        elif len(org) == 1 and {*org} != nodes:
            # If only 1 nodes and not same, False
            return False
        
        N = len(org)
        # Indegree is number of ancestor to cur nodes
        graph, indegree = defaultdict(list), [0] * (N + 1)
        
        # Compose graph
        for seq in seqs:
            for i in range(len(seq) - 1):
                s, t = seq[i], seq[i + 1]
                graph[s].append(t)
                indegree[t] += 1

        # Only enque one node when no indegree
        ans = []
        q = deque()
        for i in range(len(indegree)):
            if i > 0 and not indegree[i]:
                q.appendleft(i)
        while q:
            # If there are multiple indegree == 0, it means the supersequence
            # is not unique
            if len(q) > 1:
                return False
            
            # Pop the current source, reduce indegree of all nei, pick the 1 become 0
            s = q.pop()
            ans.append(s)
            for t in graph[s]:
                indegree[t] -= 1
                if indegree[t] == 0:
                    q.appendleft(t)

        return ans == org
```


### [269. Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        graph = defaultdict(set)
        # Trick: Control the default value in Counts
        indegree = Counter({c: 0 for word in words for c in word})
        
        # 1. Build the graph
        # Trick: Since order in one word doesn't reflect the order,
        #   you have to get the order from overlaping chs in two words
        # Trick: Use zip() to loop through 2 adjacent words, last words will be ommited
        for first_w, second_w in zip(words, words[1:]):
            for first_c, second_c in zip(first_w, second_w):
                # Don't create edge for same node
                if first_c != second_c:
                    if second_c not in graph[first_c]:
                        graph[first_c].add(second_c)
                        indegree[second_c] += 1
                    # Only look at the first different ch at 2 words, following can be
                    # any order
                    break
            else:
                if len(second_w) < len(first_w):
                    return ""
                
        # 2. Collect nodes with indegree=0, using BFS
        ans = []
        q = deque([c for c in indegree if indegree[c] == 0])
        while q:
            c = q.pop()
            ans.append(c)
            for nei in graph[c]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.appendleft(nei)
                    
        # If output nodes is less than all, there are cycles
        # node in cycle never got indegree=0, from begin to end
        if len(ans) < len(indegree):
            return ""
        
        return "".join(ans)
```