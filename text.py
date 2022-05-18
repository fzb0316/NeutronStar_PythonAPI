import Graph
import GCN_CPU
import GIN_CPU
import GCN
import GIN

from datetime import datetime
cfgFile = input('please input the cfgFile name: ')

start = datetime.now() 
graph = Graph.Graph()

graph.config._readFromCfgFile(cfgFile)
if(graph.partition_id == 0):
    graph.config._print()

graph.replication_threshold = graph.config.repthreshold
graph.load_directed(graph.config.edge_file, graph.config.vertices)

graph.generate_backward_structure()
iterations = graph.config.epochs

if(graph.config.algorithm == "GCNCPU"):
    ntsGCN = GCN_CPU.GCN_CPU(graph, iterations, False, False)
    ntsGCN._init_graph()
    ntsGCN._init_nn()
    ntsGCN._run()
elif(graph.config.algorithm == "GINCPU"):
    ntsGIN = GIN_CPU.GIN_CPU(graph, iterations, False, False)
    ntsGIN._init_graph()
    ntsGIN._init_nn()
    ntsGIN._run()
elif(graph.config.algorithm == "GCN"):
    ntsGCN = GCN.GCN(graph, iterations, False, False)
    ntsGCN._init_graph()
    ntsGCN._init_nn()
    ntsGCN._run()
elif(graph.config.algorithm == "GINGPU"):
    ntsGIN = GIN.GIN(graph, iterations, False, False)
    ntsGIN._init_graph()
    ntsGIN._init_nn()
    ntsGIN._run()


after_init = datetime.now()

end = datetime.now()

print(start)
print(after_init)
print(end)