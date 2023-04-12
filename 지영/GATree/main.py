import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
from queue import PriorityQueue

# 데이터 파일 로드
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    cities = data.to_numpy()
    return cities

# k-means 군집화 수행
class KMeansCluster:
    def __init__(self, k, cities):
        self.k = k
        self.cities = cities

    def cluster(self):
        kmeans = KMeans(n_clusters=self.k, random_state=42).fit(self.cities)
        return kmeans.labels_

class GeneticAlgorithm:
    def __init__(self, cities, population_size, mutation_rate, generations):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    # 하나의 개체 = 하나의 방문 순서
    def create_individual(self):
        individual = np.arange(0, len(self.cities)) # 첫번째 도시는 0번째 도시로 고정함
        np.random.shuffle(individual[1:])
        return individual

    def create_population(self):
        return np.array([self.create_individual() for _ in range(self.population_size)])

    def fitness(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += np.linalg.norm(self.cities[individual[i]] - self.cities[individual[i + 1]])
        total_distance += np.linalg.norm(self.cities[individual[-1]] - self.cities[0])  # Return to start
        return -total_distance

    def rank_population(self, population):
        fitness_results = np.array([self.fitness(individual) for individual in population])
        return list(np.argsort(fitness_results))

    def selection(self, population, ranked_population):
        selected_indices = []
        for i in range(len(ranked_population)):
            if random.random() < (i + 1) / len(ranked_population):
                selected_indices.append(ranked_population[i])
        return population[selected_indices]
    
    def bfs_subtree(self, individual, start_node):
        queue = [start_node]
        subtree = {start_node}  # 시작 노드는 subtree에 포함
        while queue:
            node = queue.pop(0)
            for neighbor in individual:
                if neighbor not in subtree and neighbor not in queue:
                    queue.append(neighbor)
                    subtree.add(neighbor)
        return subtree

    
    # 교차 과정: BFS 활용
    def crossover(self, parent1, parent2):
        child = np.zeros(len(parent1), dtype=int)
        child[0] = parent1[0]  # 첫 번째 노드는 항상 고정

        parent1_subtree = self.bfs_subtree(parent1, child[0])
        parent2_subtree = self.bfs_subtree(parent2, child[0])

        for i in range(1, len(parent1)):
            if parent1[i] in parent1_subtree:
                child[i] = parent1[i]
            elif parent1[i] in parent2_subtree:
                child[i] = parent2[parent1.tolist().index(parent1[i])]

        return child

    # 변이 방식: 더블 브리지
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            size = len(individual)
            pos1 = random.randint(1, size//4)
            pos2 = random.randint(pos1+1, size//2)
            pos3 = random.randint(pos2+1, 3*size//4)
            pos4 = random.randint(pos3+1, size-1)
            
            p1 = individual[:pos1]
            p2 = individual[pos2:pos3]
            p3 = individual[pos4:]
            p4 = individual[pos1:pos2]
            p5 = individual[pos3:pos4]
            
            individual = np.concatenate((p1, p2, p3, p4, p5))
        return individual


    def run(self):
        population = self.create_population()

        for generation in range(self.generations):
            print(f"Generation {generation + 1}")

            ranked_population = self.rank_population(population)
            selected_population = self.selection(population, ranked_population)

            children = []
            while len(children) < len(population):
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                children.append(child)
            population = np.array(children)

        best_individual_index = self.rank_population(population)[0]
        best_individual = population[best_individual_index]
        best_individual_distance = -self.fitness(best_individual)

        return best_individual.reshape(-1), best_individual_distance

class AStarHeuristic:
    def __init__(self, cities, paths):
        self.cities = cities
        self.paths = paths

    def euclidean_distance(self, city1, city2):
        return np.linalg.norm(self.cities[city1] - self.cities[city2])

    def reconstruct_path(self, came_from, start, goal):
        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def run(self, goal):
        start = 0
        open_set = PriorityQueue()
        open_set.put((0, start))

        came_from = {}
        g_score = {city: float('inf') for city in range(len(self.cities))}
        g_score[start] = 0
        f_score = {city: float('inf') for city in range(len(self.cities))}
        f_score[start] = self.euclidean_distance(start, goal)

        open_set_hash = {start}

        while not open_set.empty():
            current = open_set.get()[1]
            open_set_hash.remove(current)

            if current == goal:
                return self.reconstruct_path(came_from, start, goal), g_score[goal]

            for neighbor in self.paths[current]:
                tentative_g_score = g_score[current] + self.euclidean_distance(current, neighbor)

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.euclidean_distance(neighbor, goal)

                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        return None, float('inf')

class TSP:
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.cities = load_data(self.data_filename)

    def solve(self):
        k = 5
        k_means = KMeansCluster(k, self.cities)
        cluster_labels = k_means.cluster()

        # 각 군집에 대해 유전 알고리즘을 실행, 결과 저장
        cluster_results = []
        for i in range(k):
            cluster_cities = self.cities[cluster_labels == i]
            ga = GeneticAlgorithm(cluster_cities, population_size=50, mutation_rate=0.1, generations=1)
            best_individual, best_individual_distance = ga.run()
            cluster_results.append((best_individual, best_individual_distance))

        # 나눠진 군집은 A* 알고리즘을 사용하여 merge
        merged_path, merged_distance = self.merge_paths(cluster_results)

        return merged_path, merged_distance
    
    # TSP 문제의 최종 경로를 합치는 과정
    def merge_paths(self, paths):
        paths = sorted(paths, key=lambda x: x[1])  # 경로를 거리가 짧은 순서로 정렬
        start = paths.pop(0)  # start에서 가장 짧은 경로를 찾아 이어 붙임

        while paths:
            closest_path, closest_distance = None, float('inf')
            closest_index = -1
            for i, (path, distance) in enumerate(paths):
                # 가장 가까운 경로를 찾는 알고리즘은 A star 휴리스틱 사용
                a_star = AStarHeuristic(self.cities, {start[0][-1]: [path[0]]})  # 여기를 수정
                new_path, new_distance = a_star.run(path[0])  # 여기를 수정

                if new_distance < closest_distance:
                    closest_path, closest_distance = path, new_distance
                    closest_index = i

            # start와 가장 가까운 군집 경로를 결합
            paths.pop(closest_index)
            start = (np.concatenate([start[0], closest_path]), start[1] + closest_distance)

        return start



    def display_result(self, path, distance):
        print("최단 경로: ", path)
        print("총 거리: ", distance)

# 데이터 로드, TSP solve() 호출
tsp = TSP("2023_AI_TSP.csv")
final_path, final_distance = tsp.solve()

tsp.display_result(final_path, final_distance)

# 최단 경로 저장
with open('example_solution.csv', 'w+') as f:
    for city in final_path:
        f.write(f"{city}\n")