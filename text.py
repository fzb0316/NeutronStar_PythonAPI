import Graph
import GCN_CPU

print("import finish")

graph = Graph.Graph()

print("after init Graph")

graph.config._readFromCfgFile("gcn_cora.cfg")

if(graph.partition_id == 0):
    graph.config._print()

graph.replication_threshold = graph.config.repthreshold
graph.load_directed(graph.config.edge_file, graph.config.vertices)

graph.generate_backward_structure()

iterations = graph.config.epochs

print("after init Graph")

ntsGCN = GCN_CPU.GCN_CPU(graph, iterations, False, False)

ntsGCN._init_graph()
ntsGCN._init_nn()
ntsGCN._run()

print("GCN FINISHED!!!!!!!!!!!!!!!!")

