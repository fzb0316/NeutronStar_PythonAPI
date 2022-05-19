#include"../toolkits/GGNN_CPU.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(GGNN_CPU, m) {
    py::class_<GGNN_CPU_impl>(m, "GGNN_CPU")
        // .def(py::init<Graph<Empty> *, int>())
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new GGNN_CPU_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &GGNN_CPU_impl::init_graph, "init the graph data")
        .def("_init_nn", &GGNN_CPU_impl::init_nn,"init the neutral network")
        .def("_Test", &GGNN_CPU_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &GGNN_CPU_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))
        .def("_EdgeForward", &GGNN_CPU_impl::EdgeForward, "   ", py::arg("ei"))

        .def("_Loss", &GGNN_CPU_impl::Loss, "init a loss function")
        .def("_update", &GGNN_CPU_impl::Update, "update")
        .def("_run", &GGNN_CPU_impl::run, "run the GCN_CPU")
        .def("_Forward", &GGNN_CPU_impl::Forward, "forward")
        .def("_DEBUGINFO", &GGNN_CPU_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &GGNN_CPU_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &GGNN_CPU_impl::get_layersize, "  ")
        .def("print_loss", &GGNN_CPU_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &GGNN_CPU_impl::graph)
        .def_readwrite("iterations", &GGNN_CPU_impl::iterations)
        .def_readwrite("gt", &GGNN_CPU_impl::gt)
        .def_readwrite("cp", &GGNN_CPU_impl::cp)
        .def_readwrite("X", &GGNN_CPU_impl::X)
        .def_readwrite("Y", &GGNN_CPU_impl::Y)
        .def_readwrite("subgraphs", &GGNN_CPU_impl::subgraphs)
        .def_readwrite("drpmodel", &GGNN_CPU_impl::drpmodel)
        .def_readwrite("exec_time", &GGNN_CPU_impl::exec_time)
        .def_readwrite("loss", &GGNN_CPU_impl::loss);
}