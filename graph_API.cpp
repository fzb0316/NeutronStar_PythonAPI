#include "core/graph.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(Graph, m) {
    py::class_<Graph<Empty>> (m, "Graph");
        // .def(py::init<>())
//         .def_readwrite("partition_id", &Graph<Empty>::partition_id)
//         .def_readwrite("replication_threshold", &Graph<Empty>::replication_threshold)

//         .def("_init_message_map_amount", &Graph<Empty>::init_message_map_amount, "     ")
//         .def("_init_message_buffer", &Graph<Empty>::init_message_buffer, "    ")
//         .def("_init_communicatior", &Graph<Empty>::init_communicatior, "     ")
//         .def("_init_rtminfo", &Graph<Empty>::init_rtminfo, "    ")
//         .def("_init_gnnctx", &Graph<Empty>::init_gnnctx, "     ")
//         .def("_get_socket_id", &Graph<Empty>::get_socket_id, "   ")
//         .def("_get_socket_offset", &Graph<Empty>::get_socket_offset, "   ")
        
//         .def("load_directed", &Graph<Empty>::load_directed, "   " )
            
//         .def("generate_backward_structure", &Graph<Empty>::generate_backward_structure, "   " );

   py::class_<InputInfo> (m, "InputInfo");
//         .def(py::init<>())
//        // .def("_readFromCfgFile", &InputInfo::readFromCfgFile, "read paramiter from the .cfg file")
//         //.def("_print", &InputInfo::print,"   ")
//         .def_readwrite("epochs", &InputInfo::epochs)
//         .def_readwrite("repthreshold", &InputInfo::repthreshold)
//         .def_readwrite("algorithm", &InputInfo::algorithm)
//         .def_readwrite("edge_file", &InputInfo::edge_file)
//         .def_readwrite("vertices", &InputInfo::vertices);

        
}