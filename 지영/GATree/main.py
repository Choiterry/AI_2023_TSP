import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
from queue import PriorityQueue

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    cities = data.to_numpy()
    return cities

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

    def create_individual(self):
        individual = np.arange(0, len(self.cities))
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

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child = np.zeros(len(parent1), dtype=int)
        child[:crossover_point] = parent1[:crossover_point]

        next_pos = crossover_point
        for gene in parent2:
            if gene not in child:
                child[next_pos] = gene
                next_pos += 1
        return child

    def mutate(self, individual):
        for i in range(1, len(individual)):
            if random.random() < self.mutation_rate:
                swap_index = random.randint(1, len(individual) - 1)
                individual[i], individual[swap_index] = individual[swap_index], individual[i]
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

        return best_individual, best_individual_distance

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

        # Perform genetic algorithm for each cluster and store results
        cluster_results = []
        for i in range(k):
            cluster_cities = self.cities[cluster_labels == i]
            ga = GeneticAlgorithm(cluster_cities, population_size=50, mutation_rate=0.1, generations=100)
            best_individual, best_individual_distance = ga.run()
            cluster_results.append((best_individual, best_individual_distance))

        # Merge cluster paths using A* algorithm
        merged_path, merged_distance = self.merge_paths(cluster_results)

        return merged_path, merged_distance

    def merge_paths(self, paths):
        paths = sorted(paths, key=lambda x: x[1])  # Sort by distance
        start = paths.pop(0)  # Start from shortest path

        while paths:
            closest_path, closest_distance = None, float('inf')
            for i, (path, distance) in enumerate(paths):
                a_star = AStarHeuristic(self.cities, {start[0][-1]: [path[0]]})
                new_path, new_distance = a_star.run(path[0])

                if new_distance < closest_distance:
                    closest_path, closest_distance = i, new_distance

            closest_path, _ = paths.pop(closest_path)
            start = (np.concatenate([start[0], closest_path[1:]]), start[1] + closest_distance)

        return start

    def display_result(self, path, distance):
        print("최단 경로: ", path)
        print("총 거리: ", distance)

# Load data and solve TSP
tsp = TSP("2023_AI_TSP.csv")
final_path, final_distance = tsp.solve()

tsp.display_result(final_path, final_distance)