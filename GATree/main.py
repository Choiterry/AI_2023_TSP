# 유전 알고리즘과 Tree를 결합한 TSP 문제 해결

# A* 알고리즘으로 각 도시 사이의 최단 경로를 미리 계산하여 사용함.
# 이를 위해 각 도시에서 출발하여 모든 도시를 방문한 후 다시 출발 도시로 돌아오는 최단 경로를 계산
# 유전 알고리즘에서는 각 경로의 적합도를 평가할 때, 경로를 이루는 각 도시 쌍 사이의 최단 경로를 사용
# 엘리트 선택, 교차 및 돌연변이 과정은 그대로 사용하도록 함

# --------------------------------
# 최단 거리: 
# 실행 시간: 
# --------------------------------

# 휴리스틱 정의: 방법1. 현재 도시에서 가장 가까운 도시까지의 거리를 구한 후, 
# 그 다음으로 가까운 도시까지의 거리, 이후에는 계속해서 가장 가까운 도시까지의 거리를 합산한 값을 반환
# 방법2. MST(최소 신장 트리)로 정의

import csv
import random
import math
import time
import heapq

# 파일에서 도시 위치 불러오기
with open('2023_AI_TSP.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 첫번째 행 건너뛰기
    locations = [(float(row[0]), float(row[1])) for row in reader]

# 함수 정의: 두 도시 사이의 거리 계산
def get_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Nearest Neighbor 휴리스틱 함수 정의
def nearest_neighbor_heuristic(cities):
    unvisited_cities = set(range(1, len(cities)))
    current_city = 0
    path = [current_city]
    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: get_distance(cities[current_city], cities[city]))
        unvisited_cities.remove(nearest_city)
        path.append(nearest_city)
        current_city = nearest_city
    return path

# A* 알고리즘으로 TSP 문제 풀기
def a_star_algorithm(cities, heuristic_func):
    num_cities = len(cities)
    # initial_path = nearest_neighbor_heuristic(cities) + [0]  # 초기 해집합 생성
    initial_path = mst_heuristic(cities) + [0]  # 초기 해집합 생성
    initial_cost = sum(get_distance(cities[initial_path[i]], cities[initial_path[(i+1)%num_cities]]) for i in range(num_cities))
    queue = [(initial_path, initial_cost, 0)]
    visited_states = set()
    while queue:
        path, cost, depth = queue.pop(0)
        if depth == num_cities - 1:
            return path, cost
        visited_states.add(tuple(path))
        current_city = path[-2]
        for next_city in set(range(num_cities)) - set(path):
            new_path = path[:-1] + [next_city, 0]
            if tuple(new_path) not in visited_states:
                new_cost = cost + get_distance(cities[current_city], cities[next_city]) + heuristic_func(cities[new_path[:-1]])
                queue.append((new_path, new_cost, depth+1))
        queue.sort(key=lambda x: x[1])
    return None, None  # 경로가 없을 경우


# 유전 알고리즘으로 TSP 문제 풀기
def genetic_algorithm(cities, population_size=100, num_generations=1000, elite_size=20, mutation_rate=0.01):
    num_cities = len(cities)
    population = []
    for i in range(population_size):
        path = list(range(num_cities))
        random.shuffle(path)
        population.append(path)

    for i in range(num_generations):
        # 적합도 평가
        fitness_scores = []
        for j in range(population_size):
            path = population[j]
            distance = sum(get_distance(cities[path[i]], cities[path[(i+1)%num_cities]]) for i in range(num_cities))
            fitness_scores.append(1/distance)

        # 엘리트 선택
        elite_indices = sorted(range(population_size), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
        elite_population = [population[i] for i in elite_indices]

        # 교차
        new_population = elite_population[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(elite_population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        # 돌연변이
        for j in range(elite_size, population_size):
            if random.random() < mutation_rate:
                mutate(new_population[j])

        # 다음 세대로 진행
        population = new_population

    # 최적 경로 탐색
    best_path = elite_population[0]
    best_distance = sum(get_distance(cities[best_path[i]], cities[best_path[(i+1)%num_cities]]) for i in range(num_cities))

    return best_path, best_distance

# 교차 함수
def crossover(parent1, parent2):
    num_cities = len(parent1)
    start = random.randint(0, num_cities - 1)
    end = random.randint(start + 1, num_cities)
    child1 = parent1[:start] + parent2[start:end] + parent1[end:]
    child2 = parent2[:start] + parent1[start:end] + parent2[end:]
    return child1, child2

# 돌연변이 함수
def mutate(path):
    num_cities = len(path)
    i, j = random.sample(range(num_cities), 2)
    path[i], path[j] = path[j], path[i]

# 실행 시간 측정 시작
start_time = time.time()

best_path, best_distance = genetic_algorithm(locations)

end_time = time.time()

print("최적 경로:")
for i in range(len(best_path)):
    x, y = locations[best_path[i]]
    print(f"{i+1}: ({x}, {y})")

print("최단 거리:", best_distance)
print("실행 시간:", end_time - start_time, "초")