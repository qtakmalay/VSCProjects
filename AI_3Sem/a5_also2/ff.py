# Ford-Fulkerson algorithm in Python

class Graph:

    def __init__(self, graph):
        self.graph = graph  # original graph
        self.residual_graph = [[cell for cell in row] for row in graph]  # cloned graph
        self.latest_augmenting_path = [[0 for _ in row] for row in graph]  # empty graph with same dimension as graph
        self.current_flow = [[0 for _ in row] for row in graph]  # empty graph with same dimension as graph

    def bfs(self, src, snk, parent):
        visited = [False] * len(self.residual_graph)
        queue = []
        queue.append(src)
        visited[src] = True
        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.residual_graph[u]):
                if (visited[ind] == False and val > 0):
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
        return True if visited[snk] else False

    def ff_step(self, source, sink):
        """
        Perform a single flow augmenting iteration from source to sink. Update the latest augmenting path, the residual
        graph and the current flow by the maximum possible amount, according to the path found by BFS.
        @param source the source's vertex id
        @param sink the sink's vertex id
        @return the amount by which the flow has increased.
        """
        parent = [-1] * len(self.graph)
        max_flow = 0

        while (self.bfs(source, sink, parent)):
            p_flow = float("Inf")
            s = sink
            while(s != source):
                p_flow = min(p_flow, self.residual_graph[parent[s]][s])
                s = parent[s]

            self.latest_augmenting_path = [[0 for _ in row] for row in self.graph]

            v = sink
            while(v != source):
                u = parent[v]
                self.residual_graph[u][v] -= p_flow
                self.residual_graph[v][u] += p_flow

                self.latest_augmenting_path[u][v] = p_flow
                v = parent[v]

            max_flow += p_flow

        for i in range(len(self.graph)):
            for j in range(len(self.graph)):
                if self.residual_graph[i][j] < self.graph[i][j] and self.graph[i][j] > 0:
                    self.current_flow[i][j] = self.graph[i][j] - self.residual_graph[i][j]

        return max_flow
    
    def ford_fulkerson(self, source, sink):
        """
        Execute the ford-fulkerson algorithm (i.e., repeated calls of ff_step())
        @param source the source's vertex id
        @param sink the sink's vertex id
        @return the max flow from source to sink
        """
        max_flow = 0
        stopper = False
        while True:
            flow = self.ff_step(source, sink)
            if flow == 0:
                break
            max_flow += flow
        return max_flow
