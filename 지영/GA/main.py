import numpy as np
import pandas as pd
import random

# 데이터 파일 로드
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    cities = data.to_numpy()
    return cities

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
        total_distance += np.linalg.norm(self.cities[individual[-1]] - self.cities[0])
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
        child = np.zeros(len(parent1), dtype=int)
        child[0] = parent1[0]

        parent1_genes = set(parent1[1:])
        parent2_genes = set(parent2[1:])

        common_genes = parent1_genes.intersection(parent2_genes)

        for i in range(1, len(parent1)):
            if parent1[i] in common_genes:
                child[i] = parent1[i]
            else:
                child[i] = parent2[parent1.tolist().index(parent1[i])]

        return child

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

def display_result(path, distance):
    print("최단 경로: ", path)
    print("총 거리: ", distance)

# 데이터 로드 및 유전 알고리즘 실행
data_filename = "2023_AI_TSP.csv"
cities = load_data(data_filename)

ga = GeneticAlgorithm(cities, population_size=50, mutation_rate=0.1, generations=100)
final_path, final_distance = ga.run()

display_result(final_path, final_distance)

# 최단 경로 저장
with open('example_solution.csv', 'w+') as f:
    for city in final_path:
        f.write(f"{city}\n")
