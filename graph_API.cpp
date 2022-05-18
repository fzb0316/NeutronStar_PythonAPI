#include "core/graph.hpp"
#include "core/AutoDiff.h"
#include "core/gnnmini.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;


PYBIND11_MODULE(Graph, m) {

     m.def("get_time", &get_time, "get the time of system");

     py::class_<GraphOperation>(m, "Graph_Operation")
        .def(py::init([](Graph<Empty> *graph_, VertexSubset* active)
               {return new GraphOperation(graph_, active); }))
               
        .def("PropagateForwardCPU_Lockfree_multisockets", &GraphOperation::PropagateForwardCPU_Lockfree_multisockets, 
            "gather neithbour's vertex feature, the intermediate value is stored in Y", 
            py::arg("X"), py::arg("Y"), py::arg("subgraphs"));


     py::class_<nts::autodiff::ComputionPath>(m, "ComputionPath")
        .def(py::init([](GraphOperation *gt_, std::vector<CSC_segment_pinned *> subgraphs_)
               {return new nts::autodiff::ComputionPath(gt_, subgraphs_); }))
        
        .def("op_push", &nts::autodiff::ComputionPath::op_push, 
            "push the operation and intermediate result into ComputationPath, for backward propagation")
        .def("self_backward", &nts::autodiff::ComputionPath::self_backward,
            "do the backward propagation using the value that we stored while doing forward computation");

     py::class_<InputInfo> (m, "InputInfo")
          .def(py::init<>())
          .def("_readFromCfgFile", &InputInfo::readFromCfgFile, "read paramiter from the .cfg file")
          .def("_print", &InputInfo::print,"   ")

          .def_readwrite("epochs", &InputInfo::epochs)
          .def_readwrite("repthreshold", &InputInfo::repthreshold)
          .def_readwrite("algorithm", &InputInfo::algorithm)
          .def_readwrite("edge_file", &InputInfo::edge_file)
          .def_readwrite("vertices", &InputInfo::vertices);
     

     py::class_<RuntimeInfo> (m, "RuntimeInfo")
          .def(py::init<>())
          .def_readwrite("curr_layer", &RuntimeInfo::curr_layer)
          .def_readwrite("forward", &RuntimeInfo::forward)
          .def_readwrite("epoch", &RuntimeInfo::epoch);


     py::class_<Graph<Empty>> (m, "Graph")
          .def(py::init<>())
          .def_readwrite("partition_id", &Graph<Empty>::partition_id)
          .def_readwrite("replication_threshold", &Graph<Empty>::replication_threshold)
          .def_readwrite("config", &Graph<Empty>::config)
          .def_readwrite("rtminfo", &Graph<Empty>::rtminfo)

          .def("_init_message_map_amount", &Graph<Empty>::init_message_map_amount, "     ")
          .def("_init_message_buffer", &Graph<Empty>::init_message_buffer, "    ")
          .def("_init_communicatior", &Graph<Empty>::init_communicatior, "     ")
          .def("_init_rtminfo", &Graph<Empty>::init_rtminfo, "    ")
          .def("_init_gnnctx", &Graph<Empty>::init_gnnctx, "     ")
          .def("_get_socket_id", &Graph<Empty>::get_socket_id, "   ")
          .def("_get_socket_offset", &Graph<Empty>::get_socket_offset, "   ")
        
          .def("load_directed", &Graph<Empty>::load_directed, "   " )
            
          .def("generate_backward_structure", &Graph<Empty>::generate_backward_structure, "   " );


        
}