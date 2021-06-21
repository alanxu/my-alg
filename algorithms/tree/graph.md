## Tree to Graph

### [863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # Intuition: Tread tree as undirectional graph, then BFS for k levels.
        
        # Build graph
        graph = defaultdict(list)
        def connect(parent, child):
            if not child:
                return
            if parent:
                graph[parent.val].append(child.val)
                graph[child.val].append(parent.val)
            connect(child, child.left)
            connect(child, child.right)
        connect(None, root)
        
        # BFS starting from target, do it k times
        # nodes remaining in the q is the ans
        q, visited = deque([target.val]), {target.val}
        for _ in range(k):
            # Trick: Use len, each loop, the reaming
            # nodes in the q are ALL nodes of next level
            l = len(q)
            for _ in range(l):
                node = q.pop()
                for nei in graph[node]:
                    if nei not in visited:
                        q.appendleft(nei)
                        visited.add(nei)
        return list(q)
```

### [742. Closest Leaf in a Binary Tree](https://leetcode.com/problems/closest-leaf-in-a-binary-tree/)

```python
class Solution:
    def findClosestLeaf(self, root: TreeNode, k: int) -> int:
        graph, leaves = defaultdict(list), set()
        
        def build_graph(node):
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
                build_graph(node.left)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
                build_graph(node.right)
            if not node.left and not node.right:
                leaves.add(node.val)
                
        build_graph(root)
        # Trick: Use val as the keys since it is unique, save lookup time
        q = collections.deque([k])
        # Trick: Use memory to skip visited node in graph traverse
        seen = set(q)
        
        # Trick: No need to mark end of each level in this problem
        while q:
            val = q.pop()
            if val in leaves:
                return val
            for nei in graph[val]:
                if nei not in seen:
                    seen.add(nei)
                    q.appendleft(nei)
```