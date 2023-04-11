# 유전 알고리즘만을 이용한 TSP 문제 해결
# --------------------------------
# 최단 거리: 29255.70256150436
# 실행 시간: 71.83691000938416 초
# --------------------------------

import csv
import random
import math
import time
import matplotlib.pyplot as plt


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

""" 
# 모든 도시를 점으로 표시하는 그래프
x, y = zip(*locations)
plt.scatter(x, y)
plt.show()
""" 


# 유전 알고리즘으로 TSP 문제 풀기
def genetic_algorithm(cities, population_size=50, num_generations=1000, elite_size=20, mutation_rate=0.01):
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

"""
def crossover(parent1, parent2):
    num_cities = len(parent1)
    edge_dict = {}
    for i in range(num_cities):
        edge_dict[parent1[i]] = parent1[(i+1)%num_cities]
    child1, child2 = [-1]*num_cities, [-1]*num_cities
    
    # 먼저 child1에 parent1의 경로를 복사하고, 일부 간선을 parent2에서 가져와 갱신
    curr_city = parent1[0]
    child1[0] = curr_city
    while True:
        next_city = edge_dict[parent1[(parent1.index(curr_city)+1)%num_cities]]
        if next_city in parent2:
            curr_city = next_city
            child1[parent1.index(curr_city)] = curr_city
            del edge_dict[curr_city]
        else:
            break
    for i in range(num_cities):
        if child1[i] == -1:
            child1[i] = parent2[i]
    
    # child2도 마찬가지로 구성
    edge_dict = {}
    for i in range(num_cities):
        edge_dict[parent2[i]] = parent2[(i+1)%num_cities]
    curr_city = parent2[0]
    child2[0] = curr_city
    while True:
        next_city = edge_dict[parent2[(parent2.index(curr_city)+1)%num_cities]]
        if next_city in parent1:
            curr_city = next_city
            child2[parent2.index(curr_city)] = curr_city
            del edge_dict[curr_city]
        else:
            break
    for i in range(num_cities):
        if child2[i] == -1:
            child2[i] = parent1[i]

    return child1, child2
"""


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