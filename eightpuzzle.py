import heapq
import time
import random
from statistics import mean 

"""This is my first complete implementation. Everything is detailed within the notebook with benchmarks and implementation details + comments
"""



class EightPuzzle:

    def __init__(self, initial_state, goal_state):
        self.initial_state = tuple(map(tuple, initial_state)) 
        self.goal_state = tuple(map(tuple, goal_state))
        self.size = 3 

    def get_blank_position(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0: 
                    return i, j

    def get_neighbors(self, state):
        i, j = self.get_blank_position(state)  
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.size and 0 <= new_j < self.size:  
                new_state = [list(row) for row in state]  
                new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
                neighbors.append(tuple(map(tuple, new_state)))  
        return neighbors

    def is_goal(self, state):
        return state == self.goal_state

    def h1_misplaced_tiles(self, state):
        return sum(1 for i in range(self.size) for j in range(self.size) if state[i][j] != self.goal_state[i][j] and state[i][j] != 0)

    def h2_manhattan_distance(self, state):
        distance = 0
        goal = self.goal_state
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    continue
                tile_value = state[i][j]
                target_position = [(index, row.index(tile_value)) for index, row in enumerate(goal) if tile_value in row]
                if target_position:
                    target_row, target_col = target_position[0]  
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance
    
    def h3_row_column_mismatch(self, state):
        row_mismatch = 0
        column_mismatch = 0
        goal = self.goal_state
        for i in range(self.size):
            for j in range(self.size):
                tile_value = state[i][j]
                target_position = [(index, row.index(tile_value)) for index, row in enumerate(goal) if tile_value in row]
                if target_position:
                    goal_i, goal_j = target_position[0]
                    if i != goal_i:
                        row_mismatch += 1
                    if j != goal_j:
                        column_mismatch += 1
        return row_mismatch + column_mismatch  

    def h4_linear_conflict(self, state):
        manhattan_distance = self.h2_manhattan_distance(state)
        linear_conflict = 0
        for i in range(self.size):
            goal_positions = [divmod(state[i][j] - 1, self.size)[1] for j in range(self.size) if state[i][j] != 0 and divmod(state[i][j] - 1, self.size)[0] == i]
            for j in range(len(goal_positions)):
                for k in range(j + 1, len(goal_positions)):
                    if goal_positions[j] > goal_positions[k]:  
                        linear_conflict += 2
        for j in range(self.size):
            goal_positions = [divmod(state[i][j] - 1, self.size)[0] for i in range(self.size) if state[i][j] != 0 and divmod(state[i][j] - 1, self.size)[1] == j]
            for i in range(len(goal_positions)):
                for k in range(i + 1, len(goal_positions)):
                    if goal_positions[i] > goal_positions[k]:  
                        linear_conflict += 2
        return manhattan_distance + linear_conflict

    def uniform_cost_search(self):
        start_node = (self.initial_state, [], 0)  
        frontier = [(0, start_node)]  
        explored = set()  
        expanded_order = [] 
        while frontier:
            _, (state, path, cost) = heapq.heappop(frontier)  
            expanded_order.append(state)  
            if self.is_goal(state): 
                return path, expanded_order
            if state not in explored:  
                explored.add(state)
                for neighbor in self.get_neighbors(state):
                    if neighbor not in explored:
                        new_path = path + [neighbor] 
                        new_cost = cost + 1  
                        new_node = (neighbor, new_path, new_cost)
                        heapq.heappush(frontier, (new_cost, new_node))  

        return None, expanded_order  

    def best_first_search(self, heuristic):
        frontier = [(heuristic(self.initial_state), [self.initial_state])]  
        visited = set([self.initial_state])  
        expanded_nodes = 0  
        while frontier:
            _, path = heapq.heappop(frontier)  
            expanded_nodes += 1  
            current_state = path[-1] 
            if self.is_goal(current_state):  
                return path, expanded_nodes
            for neighbor in self.get_neighbors(current_state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor] 
                    heapq.heappush(frontier, (heuristic(neighbor), new_path))  

        return None, expanded_nodes  
    
    def a_star_search(self, heuristic):
        start_node = (0, [self.initial_state])  
        start_f = heuristic(self.initial_state)  
        frontier = [(start_f, start_node)]  
        visited = set([self.initial_state])  
        expanded_nodes = 0  
        while frontier:
            _, (g, path) = heapq.heappop(frontier)
            expanded_nodes += 1
            current_state = path[-1]  
            if self.is_goal(current_state):
                return path, expanded_nodes
            for neighbor in self.get_neighbors(current_state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_g = g + 1  
                    new_f = new_g + heuristic(neighbor)  
                    new_path = path + [neighbor]
                    heapq.heappush(frontier, (new_f, (new_g, new_path)))
        return None, expanded_nodes  
    
def is_solvable(puzzle):
    flat_puzzle = [tile for row in puzzle for tile in row]
    inversions = 0
    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] != 0 and flat_puzzle[j] != 0 and flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1
    return inversions % 2 == 0   

def generate_random_puzzle(goal_state):
    while True:
        puzzle = [tile for row in goal_state for tile in row]
        random.shuffle(puzzle)
        random_puzzle = [puzzle[i:i + 3] for i in range(0, len(puzzle), 3)]
        if is_solvable(random_puzzle):
            return random_puzzle

def run_benchmark(initial_state, goal_state):
    puzzle = EightPuzzle(initial_state, goal_state)
    print("Running Uniform Cost Search (UCS)...")
    start_time = time.time()
    solution_ucs, expanded_ucs_order = puzzle.uniform_cost_search()
    time_ucs = time.time() - start_time
    path_length_ucs=len(solution_ucs)
    print("\nUCS - Order of Expanded Nodes:")
    for i, node in enumerate(expanded_ucs_order):
        print(f"Step {i + 1}:")
        for row in node:
            print(row)
        print()
    print("UCS - Solution Path:")
    for i, node in enumerate(solution_ucs):
        print(f"Step {i + 1}:")
        for row in node:
            print(row)
        print()

    print(f"UCS - Path length: {len(solution_ucs)}")
    print(f"UCS - Time taken: {time_ucs:.4f} seconds")

    print("Running Best-First Search (h1: Misplaced Tiles)...")
    start_time = time.time()
    solution_bfs_h1, expanded_bfs_h1 = puzzle.best_first_search(puzzle.h1_misplaced_tiles)
    time_bfs_h1 = time.time() - start_time
    path_length_bfs_h1 = len(solution_bfs_h1)
    print("Running Best-First Search (h2: Manhattan Distance)...")
    start_time = time.time()
    solution_bfs_h2, expanded_bfs_h2 = puzzle.best_first_search(puzzle.h2_manhattan_distance)
    time_bfs_h2 = time.time() - start_time
    path_length_bfs_h2 = len(solution_bfs_h2)
    print("Running Best-First Search (h3: nb_tiles row + col)...")
    start_time = time.time()
    solution_bfs_h3, expanded_bfs_h3 = puzzle.best_first_search(puzzle.h3_row_column_mismatch)
    time_bfs_h3 = time.time() - start_time
    path_length_bfs_h3= len(solution_bfs_h3)
    print("Running Best-First Search (h4 : Manhattan + linear conflict)...")
    start_time = time.time()
    solution_bfs_h4, expanded_bfs_h4 = puzzle.best_first_search(puzzle.h4_linear_conflict)
    time_bfs_h4 = time.time() - start_time
    path_length_bfs_h4= len(solution_bfs_h4)
    print("Running A* Search (h1: Misplaced Tiles)...")
    start_time = time.time()
    solution_astar_h1, expanded_astar_h1 = puzzle.a_star_search(puzzle.h1_misplaced_tiles)
    time_astar_h1 = time.time() - start_time
    path_length_astar_h1 = len(solution_astar_h1)
    print("Running A* Search (h2: Manhattan Distance)...")
    start_time = time.time()
    solution_astar_h2, expanded_astar_h2 = puzzle.a_star_search(puzzle.h2_manhattan_distance)
    time_astar_h2 = time.time() - start_time
    path_length_astar_h2 = len(solution_astar_h2)
    print("Running A* Search  (h3 nb_tiles row +col)...")
    start_time = time.time()
    solution_astar_h3, expanded_astar_h3 = puzzle.a_star_search(puzzle.h3_row_column_mismatch)
    time_astar_h3 = time.time() - start_time
    path_length_astar_h3 = len(solution_astar_h3)
    print("Running A* Search  (h4 : Manhattan + linear conflict)...")
    start_time = time.time()
    solution_astar_h4, expanded_astar_h4 = puzzle.a_star_search(puzzle.h4_linear_conflict)
    time_astar_h4 = time.time() - start_time
    path_length_astar_h4 = len(solution_astar_h4)


    return {
        "ucs": {"time": time_ucs, "expanded": len(expanded_ucs_order), "path_length": path_length_ucs},
        "bfs_h1": {"time": time_bfs_h1, "expanded": expanded_bfs_h1, "path_length": path_length_bfs_h1},
        "bfs_h2": {"time": time_bfs_h2, "expanded": expanded_bfs_h2, "path_length": path_length_bfs_h2},
        "bfs_h3": {"time": time_bfs_h3, "expanded": expanded_bfs_h3, "path_length": path_length_bfs_h3},
        "bfs_h4": {"time": time_bfs_h4, "expanded": expanded_bfs_h4, "path_length": path_length_bfs_h4},
        "astar_h1": {"time": time_astar_h1, "expanded": expanded_astar_h1, "path_length": path_length_astar_h1},
        "astar_h2": {"time": time_astar_h2, "expanded": expanded_astar_h2, "path_length": path_length_astar_h2},
        "astar_h3": {"time": time_astar_h3,"expanded": expanded_astar_h3, "path_length": path_length_astar_h3},
        "astar_h4": {"time": time_astar_h4,"expanded": expanded_astar_h4, "path_length": path_length_astar_h4}

    }

def print_benchmark_results(results):
    print("\n--- Benchmark Results ---")

    print("Uniform Cost Search (UCS):")
    print(f"  Time: {results['ucs']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['ucs']['expanded']}")
    print(f"  Path length: {results['ucs']['path_length']}")

    print("\nBest-First Search (h1: Misplaced Tiles):")
    print(f"  Time: {results['bfs_h1']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['bfs_h1']['expanded']}")
    print(f"  Path length: {results['bfs_h1']['path_length']}")

    print("\nBest-First Search (h2: Manhattan Distance):")
    print(f"  Time: {results['bfs_h2']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['bfs_h2']['expanded']}")
    print(f"  Path length: {results['bfs_h2']['path_length']}")

    print("\nBest-First Search (h3: nb_tiles row + col):")
    print(f"  Time: {results['bfs_h3']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['bfs_h3']['expanded']}")
    print(f"  Path length: {results['bfs_h3']['path_length']}")

    print("\nBest-First Search (h4 : Manhattan + linear conflict)):")
    print(f"  Time: {results['bfs_h4']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['bfs_h4']['expanded']}")
    print(f"  Path length: {results['bfs_h4']['path_length']}") 

    print("\nA* Search (h1: Misplaced Tiles):")
    print(f"  Time: {results['astar_h1']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['astar_h1']['expanded']}")
    print(f"  Path length: {results['astar_h1']['path_length']}")

    print("\nA* Search (h2: Manhattan Distance):")
    print(f"  Time: {results['astar_h2']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['astar_h2']['expanded']}")
    print(f"  Path length: {results['astar_h2']['path_length']}")

    print("\nA* Search (h3 nb_tiles row +col):")
    print(f"  Time: {results['astar_h3']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['astar_h3']['expanded']}")
    print(f"  Path length: {results['astar_h3']['path_length']}")

    print("\nA* Search (h4 : Manhattan + linear conflict):")
    print(f"  Time: {results['astar_h4']['time']:.4f} seconds")
    print(f"  Nodes expanded: {results['astar_h4']['expanded']}")
    print(f"  Path length: {results['astar_h4']['path_length']}")

goal_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#initial_state = generate_random_puzzle(goal_state)
initial_state=tuple(map(tuple,[[1, 4,2], [0, 3, 5], [6, 7, 8]]))

print("Initial State:")
for row in initial_state:
    print(row)

results = run_benchmark(initial_state, goal_state)
print_benchmark_results(results)
