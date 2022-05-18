#include"../toolkits/COMMNET_GPU.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(COMMNET, m) {
    py::class_<COMMNET_impl>(m,"COMMNET")
        .def(py::init([](Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false)
               {return new COMMNET_impl(graph_, iterations_, false, false); }))

        .def("_init_graph", &COMMNET_impl::init_graph, "init the graph data")
        .def("_init_nn", &COMMNET_impl::init_nn,"init the neutral network")
        .def("_Test", &COMMNET_impl::Test, " test ", py::arg("s"))
        .def("_vertexForward", &COMMNET_impl::vertexForward, "   ", py::arg("a"), py::arg("x"))

        .def("_Loss", &COMMNET_impl::Loss, "init a loss function")
        //.def("_backward",)//backward is a deeper class,don't think better up to now.
        .def("_update", &COMMNET_impl::Update, "update")
        .def("_run", &COMMNET_impl::run, "run the GCN_CPU")
        .def("_Forward", &COMMNET_impl::Forward, "forward")
        .def("_DEBUGINFO", &COMMNET_impl::DEBUGINFO, "debug")
        .def("clear_gradient", &COMMNET_impl::clear_gradient, "clear the gradient in parameters and values")
        .def("get_layersize", &COMMNET_impl::get_layersize, "  ")
        .def("print_loss", &COMMNET_impl::print_loss, "print loss")
        
        .def_readwrite("graph", &COMMNET_impl::graph)
        .def_readwrite("iterations", &COMMNET_impl::iterations)
        .def_readwrite("gt", &COMMNET_impl::gt)
        .def_readwrite("cp", &COMMNET_impl::cp)
        .def_readwrite("X", &COMMNET_impl::X)
        .def_readwrite("Y", &COMMNET_impl::Y)
        .def_readwrite("subgraphs", &COMMNET_impl::subgraphs)
        .def_readwrite("drpmodel", &COMMNET_impl::drpmodel)
        .def_readwrite("exec_time", &COMMNET_impl::exec_time)
        .def_readwrite("loss", &COMMNET_impl::loss);
}
