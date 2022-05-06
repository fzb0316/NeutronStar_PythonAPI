**将NeutronStar-PythonAPI放到NeutronStarLite目录下**

**在NeutronStar中的CMakeLists.txt最后加上**

**`add_subdirectory(PythonAPI)`**

**CMakeLists.txt中写的是绝对路径，需要将cuda和libtorch更改为自己的路径**

**对源代码/NeutronStarLite/core中的graph.hpp和mpi.hpp进行了修改**

**修改之后在NeutronStar的根目录下build即可**



**将生成在/NeutronStarLite/build/PythonAPI下的Graph与GCN_CPU动态链接库与test.py、gcn_cora.cfg放入同一文件夹，运行test.py 即可**
