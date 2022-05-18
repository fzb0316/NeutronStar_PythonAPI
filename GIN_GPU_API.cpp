#include"../toolkits/GIN_GPU.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(GIN, m) {
    py::class_<GIN_impl>(m,"GIN")
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new GIN_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &GIN_impl::init_graph, "init the graph data")
        .def("_init_nn", &GIN_impl::init_nn,"init the neutral network")
        .def("_Test", &GIN_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &GIN_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))

        .def("_Loss", &GIN_impl::Loss, "init a loss function")
        //.def("_backward",)//backward is a deeper class,don't think better up to now.
        .def("_update", &GIN_impl::Update, "update")
        .def("_run", &GIN_impl::run, "run the GCN_CPU")
        .def("_Forward", &GIN_impl::Forward, "forward")
        .def("_DEBUGINFO", &GIN_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &GIN_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &GIN_impl::get_layersize, "  ")
        .def("print_loss", &GIN_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &GIN_impl::graph)
        .def_readwrite("iterations", &GIN_impl::iterations)
        .def_readwrite("gt", &GIN_impl::gt)
        .def_readwrite("cp", &GIN_impl::cp)
        .def_readwrite("X", &GIN_impl::X)
        .def_readwrite("Y", &GIN_impl::Y)
        .def_readwrite("subgraphs", &GIN_impl::subgraphs)
        .def_readwrite("drpmodel", &GIN_impl::drpmodel)
        .def_readwrite("exec_time", &GIN_impl::exec_time)
        .def_readwrite("loss", &GIN_impl::loss);
}
