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
from collections import deque

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

# 유전 알고리즘으로 TSP 문제 풀기
def genetic_algorithm(cities, population_size=100, num_generations=1000, elite_size=20, mutation_rate=0.01, subtree_size=50):
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
            child1, child2 = crossover(parent1, parent2, subtree_size)
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
def crossover(parent1, parent2, subtree_size):
    num_cities = len(parent1)
    start = random.randint(0, num_cities - 1)
    end = min(start + subtree_size, num_cities)
    subtree1 = parent1[start:end]
    subtree2 = parent2[start:end]
    new_subtree1, new_subtree2 = bfs_shuffle(subtree1[:], subtree2[:])
    child1 = parent1[:start] + new_subtree1 + parent1[end:]
    child2 = parent2[:start] + new_subtree2 + parent2[end:]
    return child1, child2


# 돌연변이 함수
def mutate(path):
    num_cities = len(path)
    i, j = random.sample(range(num_cities), 2)
    path[i], path[j] = path[j], path[i]


# BFS를 이용하여 subtree 리스트 내 도시 위치들의 순서를 랜덤하게 섞습니다.
def bfs_shuffle(subtree1, subtree2):
    queue1 = deque([subtree1])
    queue2 = deque([subtree2])
    while queue1:
        node1 = queue1.popleft()
        node2 = queue2.popleft()
        random.shuffle(node1)
        random.shuffle(node2)
        num_children = len(node1)
        if num_children > 1:
            mid = num_children // 2
            queue1.append(node1[:mid])
            queue1.append(node1[mid:])
            queue2.append(node2[:mid])
            queue2.append(node2[mid:])
    return subtree1, subtree2

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