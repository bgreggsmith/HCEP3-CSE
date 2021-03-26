#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define CSEMsg_None							0
#define CSEMsg_Cmd							1
#define CSEMsg_Flag							2
#define CSEMsg_Data							3
#define CSEMsg_WBack						4

#define CSEMsg_CfgData_T0					10
#define CSEMsg_CfgData_TStep				11
#define CSEMsg_CfgData_TEnd					12
#define CSEMsg_CfgData_nCells				13
#define CSEMsg_CfgData_nBCs					14
#define CSEMsg_CfgData_nTags				15
#define CSEMsg_CfgData_nConsts				16
#define CSEMsg_CfgData_LInterval			17
#define CSEMsg_CfgData_tagSize				18
#define CSEMsg_CfgData_masterID				19

#define CSEMsg_Data_Cell					21
#define CSEMsg_Data_Field					22
#define CSEMsg_Data_BC						23
#define CSEMsg_Data_Laplace					24
#define CSEMsg_Data_Const					25
#define CSEMsg_Data_Tag						26
#define CSEMsg_Data_Done					29

#define CSEMsg_WBack_TNow					80
#define CSEMsg_WBack_TIter					81
#define CSEMsg_WBack_FData					82
#define CSEMsg_WBack_Done					83

#define CSECmd_None							0
#define CSECmd_Cfg							110
#define CSECmd_Data							120
#define CSECmd_Ready						170
#define CSECmd_Terminate					199

#define OPMode_Inconsistent					0
#define OPMode_Configuration				10
#define OPMode_LoadData						20
#define OPMode_Ready						21
#define OPMode_Working						30
#define OPMode_Done							99

struct vtx2 {
	double x, y;
};

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

/*
 * Message format: cell data
 * 2	uint16_t	MSGType
 * 4	uint32_t	cellNumber
 * 4	int32_t		id_up
 * 4	int32_t		id_left
 * 4	int32_t		id_down
 * 4	int32_t		id_right
 * 8	double		area
 * 2	uint16_t	tags
 * 4	double		centre_x
 * 4	double		centre_y
 *
 * Total frame size: 62 bytes (Largest frame)
 * 
 * Message format: laplacian
 * 2	uint16_t	MSGType
 * 4	uint32_t	pos_i
 * 4	uint32_t	pos_j
 * 8	double		value
 * 
 * Message format: BC
 * 2	uint16_t	MSGType
 * 4	int32_t		BCNumber
 * 4	int32_t		BCType
 * 4	int32_t		BCTag
 * 8	double		BCValue
 * 2	int16_t		FieldID
 * 
 * Message format: Const
 * 2	uint16_t	MSGType
 * 2	uint16_t	Const ID
 * 8	double		Const Value
 * 
 * Message format: Field
 * 2	uint16_t	MSGType
 * 2	uint16_t	FieldID
 * 4	uint32_t	CellNumber
 * 8	double		fieldValue
 */
 
#define FrameSize_Buffer		62

int rank, nWorkers, TLSID;
int32_t opMode = 0;

int64_t nCells, nBCs, nTags, nConsts, logInterval, iter;
double TStart, TStep, TEnd, TNow;

struct state_t {
	cell_t *cellData;
	BC_t *BCData;
	double *constData;
	uint32_t *tagElems;
	uint32_t **tagCells;
	double **invLaplace;
};

union un_ui16 {
	uint16_t u16;
	int16_t i16;
	uint8_t b16[2];
};

union un_ui32 {
	uint32_t u32;
	int32_t i32;
	uint8_t b32[4];
};

union un_d64 {
	double d64;
	uint8_t b64[8];
};

void w_u32(uint32_t u, uint8_t* dat) {
	un_ui32 ui32;
	ui32.u32 = u;
	for (uint8_t i = 0; i < 4; i++) {
		dat[i] = ui32.b32[i];
	}
}

void w_d64(double d, uint8_t* dat) {
	un_d64 d64;
	d64.d64 = d;
	for (uint8_t i = 0; i < 8; i++) {
		dat[i] = d64.b64[i];
	}
}

uint16_t u16(uint8_t* dat) {
	un_ui16 ui16;
	ui16.b16[0] = dat[0];
	ui16.b16[1] = dat[1];
	return ui16.u16;
}

int16_t i16(uint8_t* dat) {
	un_ui16 ui16;
	ui16.b16[0] = dat[0];
	ui16.b16[1] = dat[1];
	return ui16.i16;
}

uint16_t u32(uint8_t* dat) {
	un_ui32 ui32;
	ui32.b32[0] = dat[0];
	ui32.b32[1] = dat[1];
	ui32.b32[2] = dat[2];
	ui32.b32[3] = dat[3];
	return ui32.u32;
}

int16_t i32(uint8_t* dat) {
	un_ui32 ui32;
	ui32.b32[0] = dat[0];
	ui32.b32[1] = dat[1];
	ui32.b32[2] = dat[2];
	ui32.b32[3] = dat[3];
	return ui32.i32;
}

double d64(uint8_t* dat) {
	un_d64 d64;
	
	for (int i=0; i<8; i++) {
		d64.b64[i] = dat[i];
	}
	
	return d64.d64;
}

state_t SimData;

void RXData() {
	MPI_Status status;
	uint8_t fBuffer[FrameSize_Buffer];
	
	int32_t MType = 0;
	
	printf("R%d: Starting data download...\n", rank);
	
	while (MType != CSEMsg_Data_Done) {
		MPI_Recv(&fBuffer, FrameSize_Buffer, MPI_BYTE, MPI_ANY_SOURCE, CSEMsg_Data, MPI_COMM_WORLD, &status);
		
		MType = u16(&fBuffer[0]);
		
		switch (MType) {
			case CSEMsg_Data_BC:{
				int32_t		BCNumber = i32(&fBuffer[2]);
				int32_t		BCType = i32(&fBuffer[6]);
				int32_t		BCTag = i32(&fBuffer[10]);
				double		BCValue = d64(&fBuffer[14]);
				int16_t		FieldID = i16(&fBuffer[22]);
				
				SimData.BCData[BCNumber].BCType = BCType;
				SimData.BCData[BCNumber].BCTag = BCTag;
				SimData.BCData[BCNumber].BCValue = BCValue;
				SimData.BCData[BCNumber].FieldID = FieldID;
				
				//printf("R%d: Set BC %d type=%d tag=%d value=%5f field=%d \n", rank, BCNumber, BCType, BCTag, BCValue, FieldID);
				} break;
				
			case CSEMsg_Data_Cell:{
				uint32_t	cellNumber = u32(&fBuffer[2]);
				SimData.cellData[cellNumber].id_up = i32(&fBuffer[6]);
				SimData.cellData[cellNumber].id_left = i32(&fBuffer[10]);
				SimData.cellData[cellNumber].id_down = i32(&fBuffer[14]);
				SimData.cellData[cellNumber].id_right = i32(&fBuffer[18]);
				SimData.cellData[cellNumber].area = d64(&fBuffer[22]);
				SimData.cellData[cellNumber].tags = i16(&fBuffer[30]);
				SimData.cellData[cellNumber].centre.x = d64(&fBuffer[32]);
				SimData.cellData[cellNumber].centre.y = d64(&fBuffer[40]);
				
				//printf("R%d: Load cell %d cx=%5f cy=%5f\n", rank, cellNumber, d64(&fBuffer[32]), d64(&fBuffer[40]));
				} break;
			
			case CSEMsg_Data_Field:{
				uint16_t	FieldID = u16(&fBuffer[2]);
				uint32_t	CellNumber = u32(&fBuffer[4]);
				SimData.cellData[CellNumber].fieldValue[FieldID] = d64(&fBuffer[8]);
			} break;
			
			case CSEMsg_Data_Laplace:{
				uint32_t i = u32(&fBuffer[2]);
				uint32_t j = u32(&fBuffer[6]);
				SimData.invLaplace[i][j] = d64(&fBuffer[10]);
			} break;
			
			case CSEMsg_Data_Const:{
				uint16_t id = u16(&fBuffer[2]);
				SimData.constData[id] = d64(&fBuffer[4]);
				printf("R%d: const[%d]=%5f\n",rank,id,SimData.constData[id]);
			} break;
			
			case CSEMsg_Data_Tag:{
				uint16_t tagID = u16(&fBuffer[2]);
				uint32_t nPos = u32(&fBuffer[4]);
				SimData.tagCells[tagID][nPos] = u32(&fBuffer[8]);
			} break;
		}
	}
	
	printf("R%d: Data download complete.\n", rank);
}

void StateAlloc(state_t *tgt) {
	tgt->cellData = new cell_t[nCells];
	tgt->BCData = new BC_t[nBCs];
	tgt->constData = new double[nConsts];
	tgt->tagElems = new uint32_t[nTags];
	tgt->tagCells = new uint32_t*[nTags];
	
	tgt->invLaplace = new double*[nCells];
	for (int i = 0; i < nCells; i ++) {
		tgt->invLaplace[i] = new double[nCells];
	}
}

//Could probably be much better with memcpy but I dont want to debug it if anything goes belly up
void CopyState(state_t *src, state_t *dst) {
	for (int i = 0; i < nCells; i++) {
		dst->cellData[i].id_up = src->cellData[i].id_up;
		dst->cellData[i].id_left = src->cellData[i].id_left;
		dst->cellData[i].id_right = src->cellData[i].id_right;
		dst->cellData[i].id_down = src->cellData[i].id_down;
		
		for (int j = 0; j < 3; j++) {
			dst->cellData[i].fieldValue[j] = src->cellData[i].fieldValue[j];
		}
		
		dst->cellData[i].tags = src->cellData[i].tags;
		dst->cellData[i].centre.x = src->cellData[i].centre.x;
		dst->cellData[i].centre.y = src->cellData[i].centre.y;
	}
	
	for (int i = 0; i < nBCs; i++) {
		dst->BCData[i].BCType = src->BCData[i].BCType;
		dst->BCData[i].BCTag = src->BCData[i].BCTag;
		dst->BCData[i].BCValue = src->BCData[i].BCValue;
		dst->BCData[i].FieldID = src->BCData[i].FieldID;
	}
	
	for (int i = 0; i < nConsts; i++) {
		dst->constData[i] = src->constData[i];
	}
	
	for (int i = 0; i < nTags; i++) {
		dst->tagElems[i] = src->tagElems[i];
		dst->tagCells[i] = new uint32_t[src->tagElems[i]];
		for (int j = 0; j < src->tagElems[i]; j++) {
			dst->tagCells[i][j] = src->tagCells[i][j];
		}
	}
	
	for (int i = 0; i < nCells; i++) {
		for (int j = 0; j < nCells; j++) {
			dst->invLaplace[i][j] = src->invLaplace[i][j];
		}
	}
}

void RXConfig() {
	MPI_Status status;
	
	printf("R%d: RXConfig()...\n", rank);
	
	MPI_Recv(&TLSID, 1, MPI_INT, MPI_ANY_SOURCE, CSEMsg_CfgData_masterID, MPI_COMM_WORLD, &status);
	
	MPI_Recv(&TStart, 1, MPI_DOUBLE, MPI_ANY_SOURCE, CSEMsg_CfgData_T0, MPI_COMM_WORLD, &status);
	MPI_Recv(&TStep, 1, MPI_DOUBLE, MPI_ANY_SOURCE, CSEMsg_CfgData_TStep, MPI_COMM_WORLD, &status);
	MPI_Recv(&TEnd, 1, MPI_DOUBLE, MPI_ANY_SOURCE, CSEMsg_CfgData_TEnd, MPI_COMM_WORLD, &status);
	MPI_Recv(&logInterval, 1, MPI_LONG, MPI_ANY_SOURCE, CSEMsg_CfgData_LInterval, MPI_COMM_WORLD, &status);
	
	MPI_Recv(&nCells, 1, MPI_LONG, MPI_ANY_SOURCE, CSEMsg_CfgData_nCells, MPI_COMM_WORLD, &status);
	MPI_Recv(&nTags, 1, MPI_LONG, MPI_ANY_SOURCE, CSEMsg_CfgData_nTags, MPI_COMM_WORLD, &status);
	MPI_Recv(&nBCs, 1, MPI_LONG, MPI_ANY_SOURCE, CSEMsg_CfgData_nBCs, MPI_COMM_WORLD, &status);
	MPI_Recv(&nConsts, 1, MPI_LONG, MPI_ANY_SOURCE, CSEMsg_CfgData_nConsts, MPI_COMM_WORLD, &status);
	
	printf("R%d: RXConfig() Allocating memory...\n", rank);
	printf("R%d: RXConfig() nCells=%d\n", rank,nCells);
	
	//Allocate memory based on received configuration
	StateAlloc(&SimData);
	
	printf("R%d: RXConfig() Zeroing inverted laplacian memory...\n", rank);
	
	//Ensure inverted laplace is filled with zeros since only the non-zero terms will be sent
	for (uint32_t i = 0; i < nCells; i++) {
		SimData.invLaplace[i] = new double[nCells];
		for (uint32_t j = 0; j < nCells; j++) {
			SimData.invLaplace[i][j] = 0.0;
		}
	}
	
	
	printf("R%d: RX tag sizing\n", rank);
	//Receive tag size information
	{
		uint8_t MType = 0;
		uint8_t fBuffer[10];
		
		while (MType == 0) {
			MPI_Recv(&fBuffer, 10, MPI_BYTE, MPI_ANY_SOURCE, CSEMsg_CfgData_tagSize, MPI_COMM_WORLD, &status);
			
			MType = fBuffer[0];
			if (MType == 0) {
				uint32_t id = u32(&fBuffer[1]);
				uint32_t size = u32(&fBuffer[5]);
				
				printf("R%d: Set tagElems[%d]=%d\n",rank, id, size);
				
				SimData.tagElems[id] = size;
				SimData.tagCells[id] = new uint32_t[size];
			}
		}
	}
	
	printf("R%d: RXConfig() done.\n", rank);
}

void CMDRouter() {
	MPI_Status status;
	int cmd;
	
	printf("R%d: CMDRouter() waiting...\n", rank);
	MPI_Recv(&cmd, 1, MPI_INT, MPI_ANY_SOURCE, CSEMsg_Cmd, MPI_COMM_WORLD, &status);
	printf("R%d: Received command code %d\n",rank, cmd);
	switch(cmd) {
		case CSECmd_Cfg:
			opMode = OPMode_Configuration;
			RXConfig();
			break;
		
		case CSECmd_Data:
			opMode = OPMode_LoadData;
			RXData();
			
			opMode = OPMode_Working;
			break;
		
		case CSECmd_Ready:
			opMode = OPMode_Working;
			break;
		
		case CSECmd_Terminate:
			opMode = OPMode_Done;
			return;
			
			break;
		
		default:
			printf("R%d: ERROR! Unknown command code %d\n", cmd);
			
			break;
	}
}

void TXBorderData(state_t *src) {
}

void TXFieldData(state_t *src) {
	//printf("R%d: TX field data at t=%5f\n", rank, TNow);
	
	//This should really use smaller frames, its completely overkill
	uint8_t fBuffer[FrameSize_Buffer];
	int CmdCode = CSEMsg_WBack;
	
	fBuffer[0] = CSEMsg_WBack_TNow;
	w_d64(TNow, &fBuffer[1]);
	
	MPI_Send(&fBuffer[0], FrameSize_Buffer, MPI_BYTE, TLSID, CSEMsg_WBack, MPI_COMM_WORLD);
	
	fBuffer[0] = CSEMsg_WBack_TIter;
	w_u32(iter, &fBuffer[1]);
	
	MPI_Send(&fBuffer[0], FrameSize_Buffer, MPI_BYTE, TLSID, CSEMsg_WBack, MPI_COMM_WORLD);
	
	fBuffer[0] = CSEMsg_WBack_FData;
	for (uint32_t i = 0; i < nCells; i++) {
		w_u32(i, &fBuffer[1]);
		w_d64(src->cellData[i].fieldValue[0], &fBuffer[5]);
		w_d64(src->cellData[i].fieldValue[1], &fBuffer[13]);
		w_d64(src->cellData[i].fieldValue[2], &fBuffer[21]);
		
		MPI_Send(&fBuffer[0], FrameSize_Buffer, MPI_BYTE, TLSID, CSEMsg_WBack, MPI_COMM_WORLD);
	}
	
	fBuffer[0] = CSEMsg_WBack_Done;
	MPI_Send(&fBuffer[0], FrameSize_Buffer, MPI_BYTE, TLSID, CSEMsg_WBack, MPI_COMM_WORLD);
}

double max(double a, double b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}

//Global alloc of things for the solver to save on memory alloc operations
uint8_t uField = 0;
uint8_t vField = 1;
uint8_t pField = 2;

double *us, *vs, *rh;

double sign(double d) {
	if (d < 0) {
		return -1;
	} else {
		return +1;
	}
}

void ApplyLimiter(state_t *tgt) {
	#pragma omp parallel for
	for (uint32_t n = 0; n < nCells; n++) {
		if (fabs(tgt->cellData[n].fieldValue[0]) > 100) {
			tgt->cellData[n].fieldValue[0] = sign(tgt->cellData[n].fieldValue[0]) * 100;
		}
		
		if (fabs(tgt->cellData[n].fieldValue[1]) > 100) {
			tgt->cellData[n].fieldValue[1] = sign(tgt->cellData[n].fieldValue[0]) * 100;
		}
		
		if (tgt->cellData[n].fieldValue[2] < 0) {
			tgt->cellData[n].fieldValue[2] = 0;
		}
		
		if (tgt->cellData[n].fieldValue[2] > 10E5) {
			tgt->cellData[n].fieldValue[2] = 10E5;
		}
	}
}

void ApplyBCs(state_t *tgt) {
	//#pragma omp parallel for
	for (uint32_t n = 0; n < nBCs; n++) {
		uint32_t z = tgt->BCData[n].BCTag;
		for (uint32_t j = 0; j < tgt->tagElems[z]; j++) {
			tgt->cellData[tgt->tagCells[z][j]].fieldValue[tgt->BCData[n].FieldID] = tgt->BCData[n].BCValue;
		}
	}
}

void Solver(state_t *rdata, state_t *wdata) {
	double nu = rdata->constData[0];
	double rho = rdata->constData[1];
	
	#pragma omp parallel for
	for (uint32_t n = 0; n < nCells; n++) {
		//ideally this would be outside but we need this inside the OpenMP scope
		//Pascals' nested subfunctions are glorious for this block of ugly
		//This should be cached
		int32_t id[4], id_xp, id_xm, id_yp, id_ym;
		double su, sv;
		uint8_t nf;
		uint8_t nx, ny;
		double u_here, v_here, dx, dy, dx1, dx2, dy1, dy2;
	
		//Populate dx, dy, neighbour info, v and u at point
		id[0] = rdata->cellData[n].id_up;
		id[1] = rdata->cellData[n].id_down;
		id[2] = rdata->cellData[n].id_left;
		id[3] = rdata->cellData[n].id_right;
		
		(id[0] >= 0) ? id_yp = id[0] : id_yp = n;
		(id[1] >= 0) ? id_ym = id[1] : id_ym = n;
		(id[3] >= 0) ? id_xp = id[3] : id_xp = n;
		(id[2] >= 0) ? id_xm = id[2] : id_xm = n;
		
		sv = 0;
		su = 0;
		nf = 0;
		#pragma omp simd
		for (uint8_t z = 0; z <= 3; z++) {
			if (id[z] >= 0) {
				su += rdata->cellData[id[z]].fieldValue[uField];
				sv += rdata->cellData[id[z]].fieldValue[vField];
				nf += 1;
			}
		}
		
		u_here = su / nf;
		v_here = sv / nf;
		
		dx = 0;
		dy = 0;
		nx = 0;
		ny = 0;
		
		if (id[2] >= 0) {
			dx1 = fabs(rdata->cellData[n].centre.x - rdata->cellData[id[2]].centre.x);
			nx += 1;
		}
		
		if (id[3] >= 0) {
			dx2 = fabs(rdata->cellData[n].centre.x - rdata->cellData[id[3]].centre.x);
			nx += 1;
		}
		
		if (id[0] >= 0) {
			dy1 = fabs(rdata->cellData[n].centre.y - rdata->cellData[id[0]].centre.y);
			ny += 1;
		}
		
		if (id[1] >= 0) {
			dy2 = fabs(rdata->cellData[n].centre.y - rdata->cellData[id[1]].centre.y);
			ny += 1;
		}
		
		(nx == 2) ? dx = 0.5 * (dx1 + dx2) : dx = max(dx1, dx2);
		(ny == 2) ? dy = 0.5 * (dy1 + dy2) : dy = max(dy1, dy2);
		
		//printf("R%dA: dx1=%5f dx2=%5f dy1=%5f dy2=%5f\n",rank, dx1, dx2, dy1, dy2);
		//printf("R%dB: n=%d dx=%5f dy=%5f uh=%5f vh=%5f ids=%d %d %d %d cx=%5f cy=%5f nx=%d ny=%d\n",rank, n, dx, dy, u_here, v_here, id[0], id[1], id[2], id[3], rdata->cellData[n].centre.x, rdata->cellData[n].centre.y, nx, ny);

//Ux discretisation
		double d2fdx2 = rdata->cellData[id_xm].fieldValue[uField] - (2 * rdata->cellData[n].fieldValue[uField]) + rdata->cellData[id_xp].fieldValue[uField];
		d2fdx2 = d2fdx2 / (dx * dx);
		
		double d2fdy2 = rdata->cellData[id_ym].fieldValue[uField] - (2 * rdata->cellData[n].fieldValue[uField]) + rdata->cellData[id_yp].fieldValue[uField];
		d2fdy2 = d2fdy2 / (dy * dy);
		
		double fdfdx = rdata->cellData[id_xp].fieldValue[uField] - rdata->cellData[id_xm].fieldValue[uField];
		fdfdx = (fdfdx / (2 * dx)) * rdata->cellData[n].fieldValue[uField];
		
		double fdfdy = rdata->cellData[id_yp].fieldValue[uField] - rdata->cellData[id_ym].fieldValue[uField];
		fdfdy = (fdfdy / (2 * dy)) * v_here;
		
		double Sum = (nu * (d2fdx2 + d2fdy2)) - (fdfdx + fdfdy);
		us[n] = rdata->cellData[n].fieldValue[uField] + (TStep * Sum);

//Uy discretisation
		d2fdx2 = rdata->cellData[id_xm].fieldValue[vField] - (2 * rdata->cellData[n].fieldValue[vField]) + rdata->cellData[id_xp].fieldValue[vField];
		d2fdx2 = d2fdx2 / (dx * dx);
		
		d2fdy2 = rdata->cellData[id_ym].fieldValue[vField] - (2 * rdata->cellData[n].fieldValue[vField]) + rdata->cellData[id_yp].fieldValue[vField];
		d2fdy2 = d2fdy2 / (dy * dy);
		
		fdfdx = rdata->cellData[id_xp].fieldValue[vField] - rdata->cellData[id_xm].fieldValue[vField];
		fdfdx = (fdfdx / (2 * dx)) * u_here;
		
		fdfdy = rdata->cellData[id_yp].fieldValue[vField] - rdata->cellData[id_ym].fieldValue[vField];
		fdfdy = (fdfdy / (2 * dy)) * rdata->cellData[n].fieldValue[vField];
		
		Sum = (nu * (d2fdx2 + d2fdy2)) - (fdfdx + fdfdy);
		vs[n] = rdata->cellData[n].fieldValue[vField] + (TStep * Sum);

//Solving the poisson equation
		double delu = -(	((rdata->cellData[id_xp].fieldValue[uField] - rdata->cellData[n].fieldValue[uField]) / dx) 
							+ ((rdata->cellData[id_yp].fieldValue[vField] - rdata->cellData[n].fieldValue[vField]) / dy) );
		
		rh[n] = (rho / TStep) * delu;
	}
	
	//The RH vector for computing the new P field is now populated, compute the matrix-vector kernel for p_n+1
	#pragma omp parallel for
	for (uint32_t n = 0; n < nCells; n++) {
		wdata->cellData[n].fieldValue[pField] = 0.0;
		#pragma omp simd
		for (uint32_t _i = 0; _i < nCells; _i++) {
			wdata->cellData[n].fieldValue[pField] += (rdata->invLaplace[n][_i] * rh[_i]);
		}
	}
	
	#pragma omp parallel for
	for (uint32_t n = 0; n < nCells; n++) {
		int32_t id[4], id_xp, id_xm, id_yp, id_ym;
		double _su, _sv;
		uint8_t _n;
		uint8_t nx, ny;
		double u_here, v_here, dx, dy, dx1, dx2, dy1, dy2;
	
		//Populate dx, dy, neighbour info, v and u at point
		id[0] = rdata->cellData[n].id_up;
		id[1] = rdata->cellData[n].id_down;
		id[2] = rdata->cellData[n].id_left;
		id[3] = rdata->cellData[n].id_right;
		
		(id[0] >= 0) ? id_yp = id[0] : id_yp = n;
		(id[1] >= 0) ? id_ym = id[1] : id_ym = n;
		(id[2] >= 0) ? id_xp = id[2] : id_xp = n;
		(id[3] >= 0) ? id_xm = id[3] : id_xm = n;
		
		_sv = 0;
		_su = 0;
		_n = 0;
		#pragma omp simd
		for (uint8_t _i = 0; _i <= 3; _i++) {
			if (id[_i] >= 0) {
				_su += rdata->cellData[id[_i]].fieldValue[uField];
				_sv += rdata->cellData[id[_i]].fieldValue[vField];
				_n += 1;
			}
		}
		
		u_here = _su / _n;
		v_here = _sv / _n;
		
		dx = 0;
		dy = 0;
		nx = 0;
		ny = 0;
		
		if (id[2] >= 0) {
			dx1 = fabs(rdata->cellData[n].centre.x - rdata->cellData[id[2]].centre.x);
			nx += 1;
		}
		
		if (id[3] >= 0) {
			dx2 = fabs(rdata->cellData[n].centre.x - rdata->cellData[id[3]].centre.x);
			nx += 1;
		}
		
		if (id[0] >= 0) {
			dy1 = fabs(rdata->cellData[n].centre.y - rdata->cellData[id[0]].centre.y);
			ny += 1;
		}
		
		if (id[1] >= 0) {
			dy2 = fabs(rdata->cellData[n].centre.y - rdata->cellData[id[1]].centre.y);
			ny += 1;
		}
		
		(nx == 2) ? dx = 0.5 * (dx1 + dx2) : dx = max(dx1, dx2);
		(ny == 2) ? dy = 0.5 * (dy1 + dy2) : dy = max(dy1, dy2);
		
		//printf("R%dB: n=%d dx=%5f dy=%5f uh=%5f vh=%5f \n",rank, n, dx, dy, u_here, v_here);
		
//Corrector:
		//Relax factor of 0.2, should be dynamic but this is a prototype
		double dfdx = (wdata->cellData[id_ym].fieldValue[pField] - wdata->cellData[n].fieldValue[pField]) / dx;
		double dfdy = (wdata->cellData[id_xm].fieldValue[pField] - wdata->cellData[n].fieldValue[pField]) / dy;
		
		wdata->cellData[n].fieldValue[uField] =  (us[n] + (TStep / rho) * dfdx);
		wdata->cellData[n].fieldValue[vField] =  (vs[n] + (TStep / rho) * dfdy);
		wdata->cellData[n].fieldValue[pField] += rdata->cellData[n].fieldValue[pField];
	}
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nWorkers);
	
	printf("R%d: MPI Initialised.\n", rank);
	
	while (opMode < OPMode_Done) {
		CMDRouter();
		
		if (opMode == OPMode_Working) {
			TNow = TStart;
			iter = 0;
			state_t *RData, *WData;
			state_t LocalData;
			
			//Ready to rock, use an MPI_Recv as a barrier to start all tasks at the same time not immediately after upload
			{
				uint8_t dummy;
				MPI_Status status;
				
				while (dummy != CSECmd_Ready) {
					MPI_Recv(&dummy, 1, MPI_BYTE, MPI_ANY_SOURCE, CSEMsg_Flag, MPI_COMM_WORLD, &status);
				}
			}
			
			//Allocate local state
			StateAlloc(&LocalData);
			
			//Copy basic data into local state
			CopyState(&SimData, &LocalData);
			
			printf("R%d: State data copied\n", rank);
			
			//Set Read & Write pointers to states for solver to operate on
			RData = &SimData;
			WData = &LocalData;
			
			//Allocate solver working memory
			us = new double[nCells];
			vs = new double[nCells];
			rh = new double[nCells];
			
			//Send initial field value back for iter=0
			TXFieldData(RData);
			
			while (TNow <= TEnd) {
				Solver(RData, WData);
				ApplyBCs(WData);
				ApplyLimiter(WData);
				
				//STUB: Send border cells to the relevant CSEs for next iteration
				TXBorderData(WData);
				
				//Increment T, iteration counter
				TNow += TStep;
				iter += 1;
				
				//Send back field data to TLS if its time for output
				if (iter % logInterval == 0) {
					TXFieldData(WData);
				}
				
				//Rotate pointers for next iteration
				{
					state_t *tmp;
					tmp = RData;
					RData = WData;
					WData = tmp;
				}
			}
			opMode = OPMode_Done;
		}
	}
	
	//Receive subdomain data
	
	printf("R%d: Finished work, shutting down...\n", rank);
	
	MPI_Finalize();
};
