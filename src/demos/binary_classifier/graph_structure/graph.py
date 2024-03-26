import random


class Node:
    def __init__(self, node_id=0, data=None):
        self.node_id = node_id
        self.data = data
        self.neighbors = []


class Graph:
    def __init__(self, indices, num_nodes=10, num_edges=9, seed=0):
        assert isinstance(num_nodes, int), "number of nodes must be an integer"
        assert type(num_edges) == int, "number of edges must be an integer"

        N = len(indices)
        random.seed(seed)

        if num_nodes > N:
            raise ValueError("number of nodes must be less than or equal to {}".format(N))
        if num_edges < num_nodes - 1:
            raise ValueError("Must have at least {} edges".format(num_nodes - 1))
        if num_edges > num_nodes*(num_nodes - 1)/2:
            raise ValueError("Must have no more than {} edges".format(num_nodes*(num_nodes-1)/2))

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_count = 0
       # random.shuffle(indices)
        self.nodes = {}

        # partition the indices into batches of size len(indices)/num_nodes and assign them to nodes
        m = int(N/num_nodes)
        for i in range(num_nodes):
            if i == num_nodes-1:
                self.nodes[i] = Node(node_id=i, data=indices[m*i:])
            else:
                self.nodes[i] = Node(node_id=i, data=indices[m*i:m*(i+1)])

        self._gen_random_graph()

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_max_degree(self):
        degrees = [len(self.nodes[idx].neighbors) for idx in self.nodes.keys()]
        return max(degrees)

    # generate a random connected graph. Inspired by https://gist.github.com/bwbaugh/4602818
    def _gen_random_graph(self):
        S, T = list(range(self.num_nodes)), []
        node_id_s = random.sample(S, 1).pop()
        S.remove(node_id_s)
        T.append(node_id_s)

        while S:
            node_id_s, node_id_t = random.sample(S, 1).pop(), random.sample(T, 1).pop()
            self._add_edge(node_id_s, node_id_t)
            S.remove(node_id_s)
            T.append(node_id_s)

        node_id_list = list(self.nodes.keys())
        while self.edge_count < self.num_edges:
            node_id_1, node_id_2 = tuple(random.sample(node_id_list, 2))
            try:
                self._add_edge(node_id_1, node_id_2)
            except ValueError:
                continue

    def _add_edge(self, node_id1, node_id2):
        if (node_id1 in self.nodes) and (node_id2 in self.nodes):
            node_1 = self.nodes[node_id1]
            node_2 = self.nodes[node_id2]
            if (node_id1 not in node_2.neighbors) and (node_id2 not in node_1.neighbors):
                node_1.neighbors.append(node_id2)
                node_2.neighbors.append(node_id1)
                self.edge_count +=1
            else:
                raise ValueError("Edge is already in the set")
        else:
            raise ValueError("One of node {} or node {} is not in the graph".format(node_id1, node_id2))


def test():
    indices = list(range(100))
    graph = Graph(indices, num_nodes=10, num_edges=9)
    test_list= []
    for i in range(10):
        print(graph.nodes[i].neighbors)
        test_list += graph.nodes[i].neighbors

    print(set(test_list))
    print(graph.edge_count)
    print(graph.nodes[9].data)


if __name__ == "__main__":
    test()
