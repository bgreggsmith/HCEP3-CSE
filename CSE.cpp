#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <omp.h>

struct cell_t {
	int32_t id_up, id_left, id_down, id_right;
	double *fieldValue;
	uint32_t *vertexIDs;
	double area;
	int32_t tags;
};

struct BC_t {
	int32_t BCType;
	int32_t BCTag;
	double BCValue;
	int32_t FieldID;
};

struct state_t {
	cell_t *cell;
	int64_t cells;
	
	uint32_t *tagElems; //Number of elements in each tag
	int32_t **tagCell; //Array of tagged element IDs for each tag
	int64_t tags;
	
	int64_t BCs;
	BC_t *BC;
	
	double t;
};

int rank, nWorkers;

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nWorkers);
	
	printf("rank=%d \n",rank);
	
	MPI_Finalize();
};
