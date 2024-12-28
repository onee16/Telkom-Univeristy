import networkx as nx
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import numpy as np

# 1. Membuat Graf Kota Bandung
graph = nx.Graph()

# Node merepresentasikan TPS dan TPA
graph.add_nodes_from(["TPS A", "TPS B", "TPS C", "TPS D", "TPS E", "TPA"])

# Edge dengan jarak antar TPS dan TPA
graph.add_weighted_edges_from([
    ("TPS A", "TPS B", 4), ("TPS A", "TPS C", 2), ("TPS B", "TPS C", 5),
    ("TPS B", "TPS D", 10), ("TPS C", "TPS E", 3), ("TPS D", "TPA", 11),
    ("TPS E", "TPS D", 4), ("TPS E", "TPA", 8)
])

# Detail kondisi TPS
tps_details = {
    "TPS A": "diangkut karena sampah penuh",
    "TPS B": "tidak diangkut karena sampah belum penuh",
    "TPS C": "diangkut karena sampah penuh",
    "TPS D": "tidak diangkut karena sampah belum penuh",
    "TPS E": "diangkut karena sampah penuh",
    "TPA": "tempat pemrosesan akhir"
}

# Menggunakan layout tetap untuk graf
fixed_pos = nx.spring_layout(graph, seed=42)  # Seed memastikan tata letak konsisten

# Visualisasi Graf
def visualize_graph(graph, path=None, title="Graf Rute Pengangkutan Sampah"):
    plt.figure(figsize=(10, 8))

    nx.draw(graph, fixed_pos, with_labels=True, node_color='lightgreen', node_size=3000, font_size=12)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, fixed_pos, edge_labels=labels)

    for node, (x, y) in fixed_pos.items():
        plt.text(x, y - 0.1, tps_details[node], fontsize=9, color='blue', ha='center')

    if path:
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, fixed_pos, edgelist=edges_in_path, edge_color='red', width=3)

    plt.title(title, fontsize=14)
    plt.show()

# Menampilkan graf awal
visualize_graph(graph, title="Graf Awal TPS ke TPA")

# 3. Algoritma A*
def a_star_route(graph, start, goal):
    path = nx.astar_path(graph, source=start, target=goal, heuristic=lambda u, v: 0, weight='weight')
    return path

# 4. Genetic Algorithm untuk optimasi rute
# Mapping nama node ke indeks
node_list = list(graph.nodes)
node_to_index = {node: idx for idx, node in enumerate(node_list)}
index_to_node = {idx: node for idx, node in enumerate(node_list)}

# Setup DEAP untuk Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(node_list)), len(node_list))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_route(individual):
    distance = 0
    for i in range(len(individual) - 1):
        node1 = index_to_node[individual[i]]
        node2 = index_to_node[individual[i + 1]]
        if graph.has_edge(node1, node2):
            distance += graph[node1][node2]['weight']
        else:
            return float('inf'),
    return distance,

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_route)

# Genetic Algorithm
population = toolbox.population(n=100)
NGEN = 50
CXPB, MUTPB = 0.7, 0.2

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, k=1)[0]
best_route = [index_to_node[idx] for idx in best_ind]

# 5. Menampilkan hasil
start, goal = "TPS A", "TPA"
a_star_path = a_star_route(graph, start, goal)

print("A* Path:", a_star_path)
print("Genetic Algorithm Path:", best_route)

# Visualisasi hasil
visualize_graph(graph, a_star_path, title="Rute A* dari TPS ke TPA")
visualize_graph(graph, best_route, title="Rute Genetic Algorithm dari TPS ke TPA")
