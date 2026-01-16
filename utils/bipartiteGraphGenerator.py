import networkx as nx
import matplotlib.pyplot as plt
import random

class GraphGenerator:
    def __init__(self, users, tasks, task_dependency_ratio, seed):
        self.users = users
        self.tasks = tasks
        self.graph = nx.Graph()
        self.edges = []
        self.task_dependency_ratio = task_dependency_ratio
        self.task_dependencies = []
        self.seed = seed
        

    def generate_graph(self):
        random.seed(self.seed)
        
        
        self.graph.add_nodes_from(self.users, bipartite=0)
        self.graph.add_nodes_from(self.tasks, bipartite=1)
        self.graph.add_nodes_from(["GNN"], bipartite=0)
        
        # Add edges with random weights and costs
        for user in self.users:
            for task in self.tasks:
                weight = random.uniform(0, 1)  # Random weight between 0 and 1
                cost = random.randint(1, 10)   # Random cost between 1 and 10
                self.graph.add_edge(user, task, weight=weight, cost=cost)
                self.edges.append((user, task, weight, cost))
        for task in self.tasks:
            weight = random.uniform(0, 0.5)  # Random weight between 0 and 1
            cost = 0 
            self.graph.add_edge("GNN", task, weight=weight, cost=cost)
            self.edges.append(("GNN", task, weight, cost))
        
        # Add dependencies between tasks
        task_dependencies = [(f"Task{i+1}", f"Task{j+1}") for i in range(int(len(self.tasks)/2)) for j in range(int(len(self.tasks)/2)+1, len(self.tasks)) if random.random() < self.task_dependency_ratio]
        self.task_dependencies = task_dependencies
        
        for t1, t2 in task_dependencies:
            index_t1 = self.tasks[int(t1[4:])-1]
            index_t2 = self.tasks[int(t2[4:])-1]
            self.graph.add_edge(index_t1, index_t2, dependency=True)

        return task_dependencies
        
    
    
    def draw_graph(self):
        pos = nx.drawing.layout.bipartite_layout(self.graph, self.users)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color=["skyblue" if n in self.users else "lightgreen" for n in self.graph.nodes()])
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={e: f"{w:.2f}" for e, w in edge_labels.items()})
        plt.title("Bipartite Graph: Users and Tasks with Weights")
        plt.show()

