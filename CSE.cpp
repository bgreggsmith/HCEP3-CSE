#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <omp.h>

struct vtx2 {
	double x, y;
}

struct cell_t {
	int32_t id_up, id_left, id_down, id_right;
	double fieldValue[3];
	double area;
	int32_t tags;
	vtx2 centre;
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
	
	int64_t consts;
	double* constValue;
	
	double t;
};

/*
 * Message format: Const data
 * 	uint16_t N_Consts
 * 		double value
 * 
 * Message format: cell data
 * 	uint16_t nFields
 * 	uint32_t N_cells
 * 		int32_t id_up
 * 		int32_t id_left
 * 		int32_t id_down
 * 		int32_t id_right
 *	 		double fieldValue
 * 		int16_t tags
 * 		double centre_x
 * 		double centre_y
 *	
 * Message format: tag data
 * 	uint16_t NTags
 * 		uint16_t NCellsInTag
 * 		uint32_t cellID
 */

int rank, nWorkers;

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nWorkers);
	
	//Receive subdomain data
	
	MPI_Finalize();
};
