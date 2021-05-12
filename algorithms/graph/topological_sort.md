
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