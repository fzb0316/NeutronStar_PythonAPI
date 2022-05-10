import Graph
import GCN_CPU

def Forward(ntsGCN):
    ntsGCN.graph.rtminfo.forward = True
    num = ntsGCN.get_layersize()
    i = 0
    for i in range(num):
        ntsGCN.graph.rtminfo.curr_layer = i
        if(i != 0):
            ntsGCN.X[i] = ntsGCN.drpmodel(ntsGCN.X[i])
        ntsGCN.gt.PropagateForwardCPU_Lockfree_multisockets(ntsGCN.X[i], ntsGCN.Y[i],ntsGCN.subgraphs)
        ntsGCN.cp.op_push(ntsGCN.X[i], ntsGCN.Y[i], 1)
        ntsGCN.X[i] = ntsGCN._vertexForward(ntsGCN.Y[i], ntsGCN.X[i])


def run_gcn(ntsGCN):
    if(ntsGCN.graph.partition_id == 0):
        print("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n" % ntsGCN.iterations)
    ntsGCN.exec_time -= GCN_CPU.get_time()
    print(ntsGCN.exec_time)

    i = 0
    for i in range(ntsGCN.iterations):
        ntsGCN.graph.rtminfo.epoch = i
        if(i != 0):
            ntsGCN.clear_gradient(i)
        # ntsGCN._Forward()
        Forward(ntsGCN)
        ntsGCN._Test(0)
        ntsGCN._Test(1)
        ntsGCN._Test(2)
        ntsGCN._Loss()
        ntsGCN.cp.self_backward(True)#need a parameter of bool,why?
        ntsGCN._update()
        if(ntsGCN.graph.partition_id == 0): 
            ntsGCN.print_loss(i)

    ntsGCN.exec_time += GCN_CPU.get_time()

if __name__ == '__main__':
    graph = Graph.Graph()
    graph.config._readFromCfgFile("gcn_cora.cfg")
    if(graph.partition_id == 0):
        graph.config._print()
    graph.replication_threshold = graph.config.repthreshold
    graph.load_directed(graph.config.edge_file, graph.config.vertices)
    graph.generate_backward_structure()
    iterations = graph.config.epochs

    ntsGCN = GCN_CPU.GCN_CPU(graph, iterations, False, False)
    ntsGCN._init_graph()
    ntsGCN._init_nn()

    # GCN_CPU::run
    run_gcn(ntsGCN)
# ntsGCN._run()