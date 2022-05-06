project(Graph)

add_subdirectory(pybind11-master) # 已经编译完的pybind11

set(PYTHONAPI_SRC
        "graph_API.cpp"
        "../comm/network.cpp"
        "../core/gnnmini.cpp"
        "../core/input.cpp"
        "../core/AutoDiff.cpp"
        "../core/bitmap.cpp"
        "../core/GraphSegment.cpp"
)

pybind11_add_module(Graph ${PYTHONAPI_SRC}) 
target_link_libraries(Graph PRIVATE ${TORCH_LIBRARIES} ${MPI_LIBRARIES} ${CUDA_LIBRARIES} cuda_propagate numa)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
../core
../comm
../cuda
${MPI_INCLUDE_PATH}
"/home/fzb/NeutronStarLite/"
"/home/fzb/NeutronStarLite/libtorch/include"
"/usr/local/cuda-11.6/include"
${TORCH_INCLUDE_DIRS}
)
