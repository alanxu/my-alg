## BFS



### [117. Populating Next Right Pointers in Each Node II](117. Populating Next Right Pointers in Each Node II)
```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root: return root
        q = collections.deque()
        q.appendleft(root)
            
        while q:
            Len = len(q)
            for i in range(Len):
                node = q.pop()
                if node.left: q.appendleft(node.left)
                if node.right: q.appendleft(node.right)
                if i < Len - 1:
                    node.next = q[-1]
                
        return root
```