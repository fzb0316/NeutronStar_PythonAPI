import Graph
import GCN_CPU
import GIN_CPU
import GCN
import GIN
import COMMNET
import GGNN_CPU
from datetime import datetime

def run_gnn(nts):
    if(nts.graph.partition_id == 0):
        print("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n" % nts.iterations)
    nts.exec_time -= Graph.get_time()
    print(nts.exec_time)

    i = 0
    for i in range(nts.iterations):
        nts.graph.rtminfo.epoch = i
        if(i != 0):
            nts.clear_gradient(i)
        nts._Forward()
        nts._Test(0)
        nts._Test(1)
        nts._Test(2)
        nts._Loss()
        nts.cp.self_backward(True)#need a parameter of bool,why?
        nts._update()
        if(nts.graph.partition_id == 0): 
            nts.print_loss(i)

    nts.exec_time += Graph.get_time()

if __name__ == '__main__':

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
        run_gnn(ntsGCN)
    elif(graph.config.algorithm == "GINCPU"):
        ntsGIN = GIN_CPU.GIN_CPU(graph, iterations, False, False)
        ntsGIN._init_graph()
        ntsGIN._init_nn()
        run_gnn(ntsGIN)
    elif(graph.config.algorithm == "GCN"):
        ntsGCN = GCN.GCN(graph, iterations, False, False)
        ntsGCN._init_graph()
        ntsGCN._init_nn()
        run_gnn(ntsGCN)
    elif(graph.config.algorithm == "GINGPU"):
        ntsGIN = GIN.GIN(graph, iterations, False, False)
        ntsGIN._init_graph()
        ntsGIN._init_nn()
        run_gnn(ntsGIN)
    elif(graph.config.algorithm == "COMMNETGPU"):
        ntsCOMMNET = COMMNET.COMMNET(graph, iterations, False, False)
        ntsCOMMNET._init_graph()
        ntsCOMMNET._init_nn()
        run_gnn(ntsCOMMNET)
    elif(graph.config.algorithm == "GGNNCPU"):
        ntsGGNN_CPU = GGNN_CPU.GGNN_CPU(graph, iterations, False, False)
        ntsGGNN_CPU._init_graph()
        ntsGGNN_CPU._init_nn()
        run_gnn(ntsGGNN_CPU)
    end = datetime.now()
    print(start)
    print(end-start)
    # ntsGCN._run()