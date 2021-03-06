#include"toolkits/GCN_CPU.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(GCN_CPU, m) {
    py::class_<GCN_CPU_impl>(m, "GCN_CPU")
        // .def(py::init<Graph<Empty> *, int>())
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new GCN_CPU_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &GCN_CPU_impl::init_graph, "init the graph data")
        .def("_init_nn", &GCN_CPU_impl::init_nn,"init the neutral network")
        .def("_Test", &GCN_CPU_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &GCN_CPU_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))

        .def("_Loss", &GCN_CPU_impl::Loss, "init a loss function")
        .def("_update", &GCN_CPU_impl::Update, "update")
        .def("_run", &GCN_CPU_impl::run, "run the GCN_CPU")
        .def("_Forward", &GCN_CPU_impl::Forward, "forward")
        .def("_DEBUGINFO", &GCN_CPU_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &GCN_CPU_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &GCN_CPU_impl::get_layersize, "  ")
        .def("print_loss", &GCN_CPU_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &GCN_CPU_impl::graph)
        .def_readwrite("iterations", &GCN_CPU_impl::iterations)
        .def_readwrite("gt", &GCN_CPU_impl::gt)
        .def_readwrite("cp", &GCN_CPU_impl::cp)
        .def_readwrite("X", &GCN_CPU_impl::X)
        .def_readwrite("Y", &GCN_CPU_impl::Y)
        .def_readwrite("subgraphs", &GCN_CPU_impl::subgraphs)
        .def_readwrite("drpmodel", &GCN_CPU_impl::drpmodel)
        .def_readwrite("exec_time", &GCN_CPU_impl::exec_time)
        .def_readwrite("loss", &GCN_CPU_impl::loss);
}
