

## Level Order Traversal
### [310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        # Alg: Level Order Traveral of Tree
        # A tree is an undirected graph in which any two vertices are connected by 
        # exactly one path. In other words, any connected graph without simple cycles 
        # is a tree.
        # If we remove the leaves of tree layer by layer, the remaining 'central' nodes
        # are what we want.
        # The alg is very similar as Topological Sorting. The difference is this is 
        # a undirectional(bidirectional) graph. And we use degree rather than indegree.
        # Everyround, we enqueue leaves (the node with degree of 1) and take them out
        # (decreasing degree of leaves parent).
        # Until the number of node poped plus numbers in queue, equals n, meaning all
        # nodes has been processed. The anwser is all nodes in queue. Which is 1 or 2.
        # because you cannot choose if the node is 1 or 2. Why it is when to stop? Cuz
        # if if the final 'centual' nodes are enqueued, others must have been poped and
        # only then the total count is n.
        # https://github.com/wisdompeak/LeetCode/tree/master/Tree/310.Minimum-Height-Trees
        if (n==1): return [0]
        if (n==2): return [0,1]
        graph, degree = defaultdict(set), [0] * (n + 1)

        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
            degree[u] += 1
            degree[v] += 1
            
        q = deque()
        for i in range(n):
            if degree[i] == 1:
                q.appendleft(i)
        
        counts = 0
        while q:
            l = len(q)
            
            for _ in range(l):
                i = q.pop()
                counts += 1
                for nei in graph[i]:
                    degree[nei] -= 1
                    if degree[nei] == 1:
                        q.appendleft(nei)
            
            if counts + len(q) == n:
                return q
        
        return []
```

## Topological Sorting
### [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

```python
class Solution:
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
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
                
        return is_possible

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Intuition: Topological Sorting - Detect circle
        # Remove courses with 0 degrees then decrease indegrees of dependent
        # courses, if finally all cources can have 0 indegree, meaning no
        # circle then we got the answer.
        graph, indegree, counts = defaultdict(set), [0] * numCourses, 0
        for a, b in prerequisites:
            graph[b].add(a)
            indegree[a] += 1
        
        q = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                q.appendleft(i)
                counts += 1
                
        while q:
            i = q.pop()
            for nei in graph[i]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.appendleft(nei)
                    counts += 1
 
        return counts == numCourses    
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

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph, indegree, counts = defaultdict(list), [0] * numCourses, 0
        ans = []
        
        for a, b in prerequisites:
            graph[b].append(a)
            indegree[a] += 1
            
        q = collections.deque()
        for i, x in enumerate(indegree):
            if x == 0:
                q.append(i)
                counts += 1
                ans.append(i)
        
        while q:
            i = q.popleft()
            for nei in graph[i]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
                    counts += 1
                    ans.append(nei)
                    
        return ans if counts == numCourses else []
```

### [444. Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/)

```python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
        # Intuition: Topological Sort
        # Get a final seq using Topological Sort and compare it with org.
        # One problem is the seq might have nodes not depends on each other
        # in this case it is not a unique seq so the answer should be False,
        # just check if there is > 1 node in q, we knwo if there is this case;
        # Other point is in this case we use graph = defaultdict(list), rather
        # than graph = defaultdict(set), because the seqs contains duplicate nodes
        # if we use set, the duplicat is filtered, but indegree still increase
        # one more time. There are two solution, one is straight forword way to
        # check duplicate and don't increase indegree more, another way is to keep
        # duplicate in list and add indegree without checking, this way is easier 
        # to program.
        
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
            # Trick: If there are multiple indegree == 0, it means the supersequence
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
                # Trick: For/Else
                # Else runs when for loop is not break
                if len(second_w) < len(first_w):
                    # If sec_w and fst_w match and fst_w is
                    # longer, it is not correct
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

### [1136. Parallel Courses](https://leetcode.com/problems/parallel-courses/)

```python
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        # Alg: Topological Sort - Use BFS, every round, add nodes with
        # indegree == 0, then remove the nodes in current round, so the nei
        # of cur round nodes can decrease their indegree by 1, and add those
        # with indegrees become 0.
        # A counts is used to track if all nodes indegree finally become zero
        # if not, means there are circle.
        # The level of the topology can be attained.
        
        # 1. Create graph and indegree
        graph, indegree = defaultdict(set), [0] * (n + 1)
        for u, v in relations:
            graph[u].add(v)
            indegree[v] += 1
        
        # 2. Use BFS
        q = deque()
        #    Use counts to count all nodes whose indegree is finally 0
        step, counts = 0, 0
        for i in range(1, n + 1):
            if indegree[i] == 0:
                q.appendleft(i)
                counts += 1
                
        while q:
            l = len(q)
            step += 1
            for _ in range(l):
                i = q.pop()
                for nei in graph[i]:
                    indegree[nei] -= 1
                    if indegree[nei] == 0:
                        q.appendleft(nei)
                        counts += 1
        
        # 3. Compare counts to detect circle                
        if counts != n:
            return -1
        
        # 4. Return step
        return step
```

### [802. Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/)

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        # Intuition: Topological Sort
        # Starting from node with 0 outdegree, they are safe. Remove
        # nodes with 0 outdegree, some other nodes have 0 outdegree,
        # they are also safe. Repeat until there is no 0-outdegree nodes.
        # If we reverse the direction of all edge, it becomes geting all
        # nodes that not in a ring
        
        # Trick: If a node's indegree eventially equals 0, it is not in
        # a ring.
        N = len(graph)
        
        # Reverse graph
        _graph, indegree = defaultdict(set), [0] * N
        for i in range(N):
            for j in graph[i]:
                _graph[j].add(i)
                indegree[i] += 1
        
        q = deque()
        for i in range(N):
            if indegree[i] == 0:
                q.appendleft(i)
            
        ans = []
        while q:
            i = q.pop()
            # If the node eventually get 0 indegree, it is SAFE
            ans.append(i)
            for nxt in _graph[i]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.appendleft(nxt)
        return sorted(ans)
        
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        # Intuition: DFS + Coloring
        # Use recursion mark each node, if node does/t have outdegree,
        # it is SAFE, if all next of a node is SAFE, the node is SAFE.
        # If any next of a node is UNSAFE, the node is Unsafe. A
        # node is UNSAFE, when it is in a circle or it's next is in a
        # circle.
        N = len(graph)
        UNKNOW, PENDING, SAFE, UNSAFE = 3, 2, 1, 0
        states = [UNKNOW] * N

        def dfs(i):
            if not graph[i]:
                states[i] = SAFE
            elif states[i] == PENDING:
                states[i] = UNSAFE
            elif states[i] == UNKNOW:
                states[i] = PENDING
                for nxt in graph[i]:
                    if dfs(nxt) == UNSAFE:
                        states[i] = UNSAFE
                        return UNSAFE
                states[i] = SAFE
            return states[i]
        
        for i in range(N):
            dfs(i)
            
        return sorted(i for i, x in enumerate(states) if x == SAFE)
```
