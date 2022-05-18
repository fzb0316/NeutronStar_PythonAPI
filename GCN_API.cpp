#include"../toolkits/GCN.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(GCN, m) {
    py::class_<GCN_impl>(m, "GCN")
        // .def(py::init<Graph<Empty> *, int>())
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new GCN_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &GCN_impl::init_graph, "init the graph data")
        .def("_init_nn", &GCN_impl::init_nn,"init the neutral network")
        .def("_Test", &GCN_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &GCN_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))

        .def("_Loss", &GCN_impl::Loss, "init a loss function")
        .def("_update", &GCN_impl::Update, "update")
        .def("_run", &GCN_impl::run, "run the GCN_CPU")
        .def("_Forward", &GCN_impl::Forward, "forward")
        .def("_DEBUGINFO", &GCN_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &GCN_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &GCN_impl::get_layersize, "  ")
        .def("print_loss", &GCN_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &GCN_impl::graph)
        .def_readwrite("iterations", &GCN_impl::iterations)
        .def_readwrite("gt", &GCN_impl::gt)
        .def_readwrite("cp", &GCN_impl::cp)
        .def_readwrite("X", &GCN_impl::X)
        .def_readwrite("Y", &GCN_impl::Y)
        .def_readwrite("subgraphs", &GCN_impl::subgraphs)
        .def_readwrite("drpmodel", &GCN_impl::drpmodel)
        .def_readwrite("exec_time", &GCN_impl::exec_time)
        .def_readwrite("loss", &GCN_impl::loss);
}
