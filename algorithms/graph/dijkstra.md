
### [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Algo: Dijkstra's Algo
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


### [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

```python
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        f = collections.defaultdict(dict)
        for a, b, p in flights:
            f[a][b] = p
        heap = [(0, src, k + 1)]
        while heap:
            p, i, k = heapq.heappop(heap)
            if i == dst:
                return p
            if k > 0:
                for j in f[i]:
                    heapq.heappush(heap, (p + f[i][j], j, k - 1))
        return -1
```

### [778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/)

```python
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # Alg: Dijkstra's Algo
        # The start point has to be defined. The cost to start point is 0.
        # The algo greedily put all the neiborgh into a heap and every
        # round picks one with least (total) cost. Usually when push a new
        # neibogh to heap, calc the total cost for the new nei and create a
        # node for heap like (total_cost_x, node_x). 
        
        # Intuition: Dijkstra's Algo
        # Starting from (0, 0), find the shortest path to (N - 1, N - 1).
        # The distance between two node is the max depth of the node in
        # the shortest path.  This is the unique challenge of this problem,
        # cuz the cost is no longer a cumulated value but a aggragated value(max).
        N = len(grid)
        heap = [(grid[0][0], 0, 0)]
        ans, seen = 0, {(0, 0)}
        while heap:
            h, r, c = heapq.heappop(heap)
            # This is the challenge of the problem.
            # Why are we sure this max value will be on the final shortest path?
            # Is it possible that we find a max value but later we didn't choose
            # the path where the node with this max value?
            # When the cur node is selected as the last of cur potential shortest path,
            # there are 2 possibilities:
            # 1. The cur node is continuing original path, because it is the smallest
            #    one in all reachable node in heap, so the height of cur node's height
            #    can >, =, < the ans (cur max), and we just need check if it is max and 
            #    update ans
            # 2. The cur node is a new path different with original path, then the height
            #    of cur node has to be greater than all nodes in original path, because
            #    otherwise it already be selected before. In this case, when path switches
            #    max(ans, h) will always reflect the cost of the new optimal path
            # So, ans will be reset everytime a new optimal path is selected, in other words,
            # ans always track the cost of cur path
            ans = max(ans, h)
            if r == c == N - 1:
                return ans
            
            for _r, _c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= _r < N and 0 <= _c < N and (_r, _c) not in seen:
                    seen.add((_r, _c))
                    heapq.heappush(heap, (grid[_r][_c], _r, _c))            
```

### [505. The Maze II](https://leetcode.com/problems/the-maze-ii/)

```python
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        # Intuition: Dijkastra's Alg
        R, C = len(maze), len(maze[0])
        heap, visited = [(0, start[0], start[1])], {(start[0], start[1]): 0}
        while heap:
            distance, r, c = heapq.heappop(heap)
            if r == destination[0] and c == destination[1]:
                return distance
            for d0, d1 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                # Trick: Stop when next is not valid
                # Roll the ball toward direction d
                _distance, _r, _c = distance, r, c
                while 0 <= _r + d0 < R and 0 <= _c + d1 < C and maze[_r + d0][_c + d1] != 1:
                    _r, _c = _r + d0, _c + d1
                    _distance += 1

                if (_r, _c) not in visited or _distance < visited[(_r, _c)]:
                    visited[(_r, _c)] = _distance
                    heapq.heappush(heap, (_distance, _r, _c))
        return -1
```