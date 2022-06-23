# Notes on how to build Megatron-LM and its depednecies for NVIDIA and AMD GPUs

This folder provides steps to run vanilla Megatron-LM adapted from NVIDIA repo for both CUDA and ROCm. In this steps the data is prepared under `/tmp/<user id>`.

## NVIDIA - CUDA build
This is the process Sam's followed to get Megatron (NVIDIA) work on top ov NVHPC 21.3. 

This was done in a AMD internal machine that runs A100, here the details:
```
NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3
```

### Build and run
Note that this machines have a LMOD hierarchy that provides the CUDA/NVHPC environment as well as numactl used to control the resource binding. Here the steps:
* clone this repo
* create miniconda instalation under `./from-amd/`
* `cd ./from-amd/cuda`
* Adjust commented function calls at the end of `build-cuda.sh` depending on whether you want to build or just run.
* `./build-cuda.sh`

This script will get the dependencies and configure environments.

The end of the log produced during the run should show:
```
-------------------------------------------------------------------------------------------------
 validation loss at iteration 19000 | lm loss value: 1.280714E+00 | lm loss PPL: 3.599210E+00 | 
-------------------------------------------------------------------------------------------------
saving checkpoint at iteration   19065 to /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
  successfully saved checkpoint at iteration   19065 to /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
calculating metrics ...
 > |epoch: 2| metrics for dev-middle: correct / total = 517 / 1436 = 36.0028 %, elapsed time (sec): 19.963
 > |epoch: 2| metrics for dev-high: correct / total = 1163 / 3451 = 33.7004 %, elapsed time (sec): 48.028
 >> |epoch: 2| overall: correct / total = 1680 / 4887 = 34.3769 %
done :-)
```

## AMD - ROCm build
This is the process Sam's followed to get Megatron (NVIDIA) work on top of ROCm 5.2.0 rc. Prior ROCm versions should work as well. 

This was done in a AMD internal machine that runs MI210, here the details:
```
*******                  
Agent 9                  
*******                  
  Name:                    gfx90a                             
  Uuid:                    GPU-a4533375d58e90fc               
  Marketing Name:                                             
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        128(0x80)                          
  Queue Min Size:          4096(0x1000)                       
  Queue Max Size:          131072(0x20000)                    
  Queue Type:              MULTI                              
  Node:                    8                                  
  Device Type:             GPU                                
  Cache Info:              
    L1:                      16(0x10) KB                        
    L2:                      8192(0x2000) KB                    
  Chip ID:                 29711(0x740f)                      
  Cacheline Size:          64(0x40)                           
  Max Clock Freq. (MHz):   1700                               
  BDFID:                   25344                              
  Internal Node ID:        8                                  
  Compute Unit:            104                                
  SIMDs per CU:            4                                  
  Shader Engines:          8                                  
  Shader Arrs. per Eng.:   1                                  
  WatchPts on Addr. Ranges:4                                  
  Features:                KERNEL_DISPATCH 
  Fast F16 Operation:      TRUE                               
  Wavefront Size:          64(0x40)                           
  Workgroup Max Size:      1024(0x400)                        
  Workgroup Max Size per Dimension:
    x                        1024(0x400)                        
    y                        1024(0x400)                        
    z                        1024(0x400)                        
  Max Waves Per CU:        32(0x20)                           
  Max Work-item Per CU:    2048(0x800)                        
  Grid Max Size:           4294967295(0xffffffff)             
  Grid Max Size per Dimension:
    x                        4294967295(0xffffffff)             
    y                        4294967295(0xffffffff)             
    z                        4294967295(0xffffffff)             
  Max fbarriers/Workgrp:   32                                 
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    67092480(0x3ffc000) KB             
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 2                   
      Segment:                 GROUP                              
      Size:                    64(0x40) KB                        
      Allocatable:             FALSE                              
      Alloc Granule:           0KB                                
      Alloc Alignment:         0KB                                
      Accessible by all:       FALSE                              
  ISA Info:                
    ISA 1                    
      Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        4294967295(0xffffffff)             
        y                        4294967295(0xffffffff)             
        z                        4294967295(0xffffffff)             
      FBarrier Max Size:       32 
```

### Build and run
Note that this machines have a LMOD hierarchy that provides the ROCm environment as well as numactl used to control the resource binding. Here the steps:
* clone this repo
* create miniconda instalation under `./from-amd/`. The same as the CUDA build can be used.
* `cd ./from-amd/rocm`
* Adjust commented function calls at the end of `build-rocm.sh` depending on whether you want to build or just run.
* `./build-rocm.sh`

This script will get the dependencies and configure environments.

The end of the log produced during the run should sbe similar to the CUDA run.
