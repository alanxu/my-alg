
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