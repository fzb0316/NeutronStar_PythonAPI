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
        .def("_Test", &GCN_CPU_impl::Test, " test ", 
                py::arg("s"))
                
        .def("_vertexForward", &GCN_CPU_impl::vertexForward, "   ", 
                py::arg("a"), py::arg("x"))

        .def("_Loss", &GCN_CPU_impl::Loss, "init a loss function")
        //.def("_backward",)//backward is a deeper class,don't think better up to now.
        .def("_update", &GCN_CPU_impl::Update, "update")
        .def("_run", &GCN_CPU_impl::run, "run the GCN_CPU")
        .def("_Forward", &GCN_CPU_impl::Forward, "forward")
        .def("_DEBUGINFO", &GCN_CPU_impl::DEBUGINFO, "debug");
}
