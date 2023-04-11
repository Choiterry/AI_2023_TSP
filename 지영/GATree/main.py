"""
군집화만 사용한 경우 - 2500 거리값이 나옴
"""
import numpy as np
import pandas as pd
import heapq
import itertools
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
class TSPNode:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y

    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class TSPSolver:
    def __init__(self, nodes, start_index=0):
        self.nodes = nodes
        self.start_index = start_index
        self.num_nodes = len(nodes)

    def _solve_subproblem(self, cluster):
        visited = set()
        start_node = self.nodes[self.start_index]
        visited.add(self.start_index)

        path = [start_node]

        while len(visited) < len(cluster):
            nearest_node = None
            min_distance = float('inf')

            for i in cluster:
                if i in visited:
                    continue

                distance = path[-1].distance(self.nodes[i])

                if distance < min_distance:
                    nearest_node = i
                    min_distance = distance

            visited.add(nearest_node)
            path.append(self.nodes[nearest_node])

        return path, visited

    def solve(self):
        clustering = DBSCAN(eps=30, min_samples=8).fit_predict([(node.x, node.y) for node in self.nodes])
        unique_clusters = np.unique(clustering)
        
        overall_path = []
        overall_visited = set()
        total_distance = 0

        for cluster_label in unique_clusters:
            cluster_indices = np.where(clustering == cluster_label)[0]
            cluster_path, visited = self._solve_subproblem(cluster_indices)
            
            overall_path.extend(cluster_path)
            overall_visited |= visited

            total_distance += sum([cluster_path[i].distance(cluster_path[i + 1]) for i in range(len(cluster_path) - 1)])

        overall_path.append(self.nodes[self.start_index])
        total_distance += overall_path[-1].distance(overall_path[-2])

        return total_distance, [node.index for node in overall_path]
    
def main():
    df = pd.read_csv('2023_AI_TSP.csv', header=None)
    nodes = [TSPNode(index, x, y) for index, (x, y) in df.iterrows()]

    solver = TSPSolver(nodes)
    total_distance, path = solver.solve()

    print("최단거리:", total_distance)
    print("방문한 노드 순서:", path)

if __name__ == "__main__":
    main()