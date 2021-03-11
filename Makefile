CXX = mpicxx
CXX_FLAGS = --std=c++17 -Wall -Wextra -march=native -O3 -g -DOMPI_SKIP_MPICXX -fopenmp
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gcc 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157

student_submission: CSE.cpp
	$(CXX) $(CXX_FLAGS) -o cse CSE.cpp

clean:
	rm -f CSE
