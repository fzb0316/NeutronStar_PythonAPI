#include"../toolkits/GIN_CPU.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(GIN_CPU, m) {

    py::class_<GIN_CPU_impl>(m,"GIN_CPU")
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new GIN_CPU_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &GIN_CPU_impl::init_graph, "init the graph data")
        .def("_init_nn", &GIN_CPU_impl::init_nn,"init the neutral network")
        .def("_Test", &GIN_CPU_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &GIN_CPU_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))

        .def("_Loss", &GIN_CPU_impl::Loss, "init a loss function")
        //.def("_backward",)//backward is a deeper class,don't think better up to now.
        .def("_update", &GIN_CPU_impl::Update, "update")
        .def("_run", &GIN_CPU_impl::run, "run the GCN_CPU")
        .def("_Forward", &GIN_CPU_impl::Forward, "forward")
        .def("_DEBUGINFO", &GIN_CPU_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &GIN_CPU_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &GIN_CPU_impl::get_layersize, "  ")
        .def("print_loss", &GIN_CPU_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &GIN_CPU_impl::graph)
        .def_readwrite("iterations", &GIN_CPU_impl::iterations)
        .def_readwrite("gt", &GIN_CPU_impl::gt)
        .def_readwrite("cp", &GIN_CPU_impl::cp)
        .def_readwrite("X", &GIN_CPU_impl::X)
        .def_readwrite("Y", &GIN_CPU_impl::Y)
        .def_readwrite("subgraphs", &GIN_CPU_impl::subgraphs)
        .def_readwrite("drpmodel", &GIN_CPU_impl::drpmodel)
        .def_readwrite("exec_time", &GIN_CPU_impl::exec_time)
        .def_readwrite("loss", &GIN_CPU_impl::loss);
}
