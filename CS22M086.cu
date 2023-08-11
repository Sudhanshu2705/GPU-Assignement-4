#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here

//Device Function to sort the array using insertion sort from start to end index
__device__ void sortArray(int *array,int start,int end){
  for(int i=start+1;i<end;i++){
    int j=i;
    while(j>start && array[j-1]>array[j]){
      int temp;
      temp = array[j];
      array[j]=array[j-1];
      array[j-1]=temp;
      j--;
    }
  }
}


//Kernel to process requests. Each thread corresponds to a facilty.
__global__ void processRequest(int totalFac,int * d_succ_reqs,int *d_capacity,int *d_workList,int *d_facReqIndex,int *d_req_slots,int *d_req_start,int *d_req_cen,int *d_success,int *d_fail){
  int thid;
  thid = blockIdx.x * blockDim.x + threadIdx.x;
  int cap;
  int start,end;
  int slots[24];
  if(thid<totalFac){
    for(int i=0;i<24;i++){
      slots[i]=0;
    }  
    cap = d_capacity[thid];
    if(thid == 0){
      start = 0;
    }
    else{
      start = d_facReqIndex[thid-1];
    }
    end = d_facReqIndex[thid];
    sortArray(d_workList,start,end);

    for(int i=start;i<end;i++){
      int f=1;
      int req_id = d_workList[i];
      int req_start = d_req_start[req_id];
      int req_slots = d_req_slots[req_id];
      int req_cen = d_req_cen[req_id];
      for(int j=req_start;j<(req_start + req_slots);++j){
        if(slots[j-1]==cap){
          f=0;
          break;
        }
      }
      if(f == 1){
        for(int k=req_start;k<(req_start + req_slots);++k){
          slots[k-1]++;
        }
        atomicAdd(&d_succ_reqs[req_cen],1);
        atomicAdd(&d_success[0],1);
      }
    }
  }
}

// Kernel to create a worklist for each faciltiy. Each thread correponds to a request.
__global__ void workList(int totalReq,int *d_centerIndex,int *d_req_cen,int *d_req_fac,int *d_workList,int *d_facReqIndex,int * d_workListIndex){
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid<totalReq){
    int req_cen;
    int req_fac;
    req_cen = d_req_cen[thid];
    req_fac = d_req_fac[thid];
    int start,position;
    if(req_cen==0)
      start = 0;
    else
      start = d_centerIndex[req_cen-1];
    
    position = start + req_fac;

    int workIndex,facReqIndex;
    
    if(position==0){
      facReqIndex = 0;
    }
    else{
      facReqIndex = d_facReqIndex[position -1];
    }
    workIndex = atomicAdd(&d_workListIndex[position],1);
    workIndex = facReqIndex + workIndex;
    d_workList[workIndex] = thid;
  }
}


// Kernel to count requets for each faciltiy. Each threads corresponds to a request.
__global__ void facReqCount(int totalReq,int *d_centerIndex, int *d_req_cen,int *d_req_fac,int *d_facReqIndex){
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if(thid<totalReq){
    int req_cen;
    int req_fac;
    req_cen = d_req_cen[thid];
    req_fac = d_req_fac[thid];
    int start,position;
    if(req_cen==0)
      start = 0;
    else
      start = d_centerIndex[req_cen-1];
    position = start + req_fac;
    atomicAdd(&d_facReqIndex[position],1);
  }
}


//Kernel to find prefix sum. Works for a block.
__global__ void prefix_sum_kernel(int start,int end, int * array){
  int tmp=0;
  int thid = start + threadIdx.x;
  int N;
  N = end - start;
  for(int off = 1;off<N;off*=2){
    if(thid >= (off + start)){
      tmp = array[thid-off];
    }
    __syncthreads();
    if(thid >= (off + start)){
      atomicAdd(&array[thid],tmp);
    }
    __syncthreads();
  }

}

// Function to calucate prefix_sum that uses prefix_sum_kernel to calculate prefix sum of any aray of any size.
void prefix_sum(int *array,int N){
  int num_block,num_threads;
  num_block = ceil((float)(N-1024)/1023)+1;
  int start,end;
  start = 0;
  for(int i=0;i<num_block;i++){
    if(i+1==num_block){
      end = N;
    }
    else{
      end = start + 1024;
    }
    num_threads = end-start;
    prefix_sum_kernel<<<1,num_threads>>>(start,end,array);
    cudaDeviceSynchronize();
    start = start + 1023;
  }  
  cudaDeviceSynchronize();
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    // char *inputfilename = "./Input/input12.txt";
    char *inputfilename =  argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre 
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		
    char *outputfilename = argv[2];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w"); 


    //*********************************
    // Call the kernels here
    int totalReq = R;
    int totalFac = k1;

    // Declaration of device memory pointers.
    int *d_req_id,*d_req_cen,*d_req_fac,*d_req_start,*d_req_slots,*d_capacity;

    // Allocating space on device.
    cudaMalloc(&d_req_id, totalReq *sizeof(int));
    cudaMalloc(&d_req_cen, totalReq *sizeof(int));
    cudaMalloc(&d_req_fac, totalReq *sizeof(int));
    cudaMalloc(&d_req_start, totalReq *sizeof(int));
    cudaMalloc(&d_req_slots, totalReq *sizeof(int));
    cudaMalloc(&d_capacity, totalFac *sizeof(int));

    // Copying necessory data on device.
    cudaMemcpy(d_req_id,req_id,totalReq * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen,req_cen,totalReq * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac,req_fac,totalReq * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start,req_start,totalReq * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots,req_slots,totalReq * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity,capacity,totalFac* sizeof(int),cudaMemcpyHostToDevice);

    // Declaration of device memory pointers to store final results.
    int *d_succ_reqs,*d_success,*d_fail;

    // Allocating space on device to store result.
    cudaMalloc(&d_succ_reqs,N * sizeof(int));
    cudaMalloc(&d_success,sizeof(int));
    cudaMalloc(&d_fail,sizeof(int));

    // Initializing devices arrays.
    cudaMemset(d_succ_reqs,0,N * sizeof(int));
    cudaMemset(d_success,0,sizeof(int));
    cudaMemset(d_fail,0,sizeof(int));

    // Declaration od device pointers to store intermediate results.
    int *d_workList,*d_facReqIndex,*d_workListIndex,*d_centerIndex;

    // Allocating space on device.
    cudaMalloc(&d_centerIndex,N*sizeof(int));
    cudaMalloc(&d_facReqIndex,totalFac * sizeof(int));
    cudaMalloc(&d_workList,totalReq * sizeof(int));
    cudaMalloc(&d_workListIndex,totalFac * sizeof(int));

    // Initializing arrays.
    cudaMemset(d_facReqIndex,0,totalFac * sizeof(int));
    cudaMemset(d_workListIndex,0,totalFac * sizeof(int));

    cudaMemcpy(d_centerIndex,facility,N*sizeof(int),cudaMemcpyHostToDevice);
    
    // CPU aaray to store final result.
    int *h_success = (int *)malloc(sizeof(int));

    // Calcuating prefix sum of array containing number of facilities of each center.
    // result helps in finding indexes of individual facilities in d_facReqIndex.
    prefix_sum(d_centerIndex,N);
    
    // Kernel call to find number of request in each facility.
    int block;
    block = ceil((float)totalReq/1024);
    facReqCount<<<block,1024>>>(totalReq,d_centerIndex,d_req_cen,d_req_fac,d_facReqIndex);
    cudaDeviceSynchronize();

    // calculating prefex sum of array containing number of requests for each facility.
    // Helps in finding index range of requets in worklist for each facility.
    prefix_sum(d_facReqIndex,totalFac);
    
    // Kernel Call to add requests in the workList.
    //Requests of a facility are added in the range of index obtained from previous prefix_sum.
    workList<<<block,1024>>>(totalReq,d_centerIndex,d_req_cen,d_req_fac,d_workList,d_facReqIndex,d_workListIndex);
    
    int num_block;
    num_block =  ceil((float)totalFac/1024);

    // Kernel Call to process the request.
    // Each thread corresponds to a faciltity.
    // Each faclity knows the range of indexes which the requests it has to process.
    // It first sorts those requests and then processes them sequentially.
    processRequest<<<num_block,1024>>>(totalFac,d_succ_reqs,d_capacity,d_workList,d_facReqIndex,d_req_slots,d_req_start,d_req_cen,d_success,d_fail);
    cudaDeviceSynchronize();
    
    // Copyint the final result from device to CPU.
    cudaMemcpy(succ_reqs,d_succ_reqs,N * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_success,d_success,sizeof(int),cudaMemcpyDeviceToHost);

    success = *h_success;
    fail = totalReq - success;

    //********************************

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n",succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}