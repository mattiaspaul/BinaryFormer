#include <iostream>
#include <vector>

#include <torch/torch.h>

//#include <torch/csrc/jit/import.h>

#include <sys/time.h>
#include <torch/script.h>

#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

//#include "bridge.h"

/*
    int64_t B = 4; int64_t C = 256; int64_t N = 256;
    auto key = .5*torch::randn({B,C,N});
    auto query = .5*torch::randn({B,C,N});
    auto result = torch::zeros({B,C,N});
    at::native::mps::multiplyTensors(query,key,result);
    auto result1 = key*query;
    std::cout << (result-result1).norm() << std::endl;


}*/

//bitwisePopulationCount
//bitwiseXOR
/*
 - (MPSGraphTensor *) bitwiseXORWithPrimaryTensor:(MPSGraphTensor *) primaryTensor
                                  secondaryTensor:(MPSGraphTensor *) secondaryTensor
                                             name:(NSString *) name;
 
 - (MPSGraphTensor *) bitwisePopulationCountWithTensor:(MPSGraphTensor *) tensor
                                                  name:(NSString *) name;
 - (MPSGraphTensor *) HammingDistanceWithPrimaryTensor:(MPSGraphTensor *) primaryTensor
                                       secondaryTensor:(MPSGraphTensor *) secondaryTensor
                                        resultDataType:(MPSDataType) resultDataType
                                                  name:(NSString *) name;
 @abstract   Create a  Hamming Distance op and return the result tensor, it supports broadcasting as well.
 *  @discussion The Hamming Distance is computed between sets of vectors and the last dimension(s) of each
 *              input tensor is considered a vector. If the shape of @ref primaryTensor is `[Na, M, K]` and the shape
 *              of @ref secondaryTensor is `[Nb, N, K]`, with Na, Nb being any batch dimensions,
 *              then the result shape is `[Na/Nb, M, N]`, where `Na/Nb` are the broadcasted batch dimensions.
 *              The result datatype is either MPSDataTypeUInt32 or MPSDataTypeUInt16.
 *
 *  @param      primaryTensor          LHS tensor of the binary Op
 *  @param      secondaryTensor      RHS tensor of the binary Op
 *  @param      resultDataType        Must be either MPSDataTypeUInt32 or MPSDataTypeUInt16
 *  @param      name                              name for the operation
 *
 *  @return     A valid MPSGraphTensor object.
 
-(MPSGraphTensor *) HammingDistanceWithPrimaryTensor:(MPSGraphTensor *) primaryTensor
                                     secondaryTensor:(MPSGraphTensor *) secondaryTensor
                                      resultDataType:(MPSDataType) resultDataType
                                                name:(NSString * _Nullable) name

 */

#include "hammingAttention.h"

int main() {
    /*hello();
      torch::Tensor tensor = torch::rand({2, 3});
      std::cout << tensor << std::endl;
    auto A = torch::randn({64,64});
    auto B = torch::randn({64,64});
    auto C = torch::zeros({64,64});
    at::native::mps::multiplyTensor(A,B,C);
    auto D = A*B;
    std::cout << (D-C).norm() << std::endl;*/
    int B = 16;
    int N = 2048;
    int D = 256;
    int Dv = 64;
    
    auto Q = torch::randn({B,N,D});
    auto K = torch::randn({B,N,D});
    auto V = torch::randn({B,N,Dv});
    //timeval time1,time2;
    auto res0 = torch::zeros({B,N,Dv});
    auto res1 = torch::zeros({B,N,Dv});
    at::native::mps::tiledSDPAfloat(Q,K,V,res1);
    std::cout<<res1.norm()<<std::endl;
    auto res2 = torch::scaled_dot_product_attention(Q,K,V);
    std::cout<<res1.norm()<<", "<<res2.norm()<<" diff: "<<(res1-res2).norm()<<std::endl;
    
    auto rand_bin = torch::cat({torch::tensor({0,1,1,0,1,0}),torch::zeros(2)}).to(torch::kUInt8).view({1,1,8});
    auto output8 = torch::zeros({1,1,1}).to(torch::kByte);
    at::native::mps::packbit8Tensor(rand_bin,output8);
    std::cout<<output8<<std::endl;
    
    timeval time1,time2;
    at::native::mps::tiledHammingAttentionReLU(Q,K,V,res0);
    int iter = 10;
    
    gettimeofday(&time1, NULL);
    for(int i=0;i<iter;i++){
        torch::NoGradGuard no_grad;
        at::native::mps::tiledHammingAttentionReLU(Q,K,V,res0);
    }
    gettimeofday(&time2, NULL);

    float time_int=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time HammingAttention ReLU:\t\t"<<time_int*1000.0f/iter<<" msecs."<<std::endl;
    at::native::mps::tiledSDPAfloatReLU(Q,K,V,res1);

    
    gettimeofday(&time1, NULL);
    for(int i=0;i<iter;i++){
        torch::NoGradGuard no_grad;
        at::native::mps::tiledSDPAfloat(Q,K,V,res1);
    }
    gettimeofday(&time2, NULL);
    float time_floats=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time SDPA softmax float:\t\t"<<time_floats*1000.0f/iter<<" msecs."<<std::endl;
    at::native::mps::tiledSDPAfloatReLU(Q,K,V,res1);

    gettimeofday(&time1, NULL);
    for(int i=0;i<iter;i++){
        torch::NoGradGuard no_grad;
        at::native::mps::tiledSDPAfloatReLU(Q,K,V,res1);
    }
    gettimeofday(&time2, NULL);
    float time_float=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time SDPA relu float:\t\t"<<time_float*1000.0f/iter<<" msecs."<<std::endl;

    
    /*
    int iter = 10;
    timeval time1,time2;

    auto rand_bin = torch::ge(torch::rand({B,N,D}),0.5).to(torch::kUInt8);
    
//    auto bit_mult = torch::pow(2,torch::arange(32)).to(torch::kInt64);
//    auto bits = (bit_mult*rand_bin.unflatten(-1,{-1,32}).to(torch::kInt64)).sum(-1);
    auto bit_mult = torch::pow(2,torch::arange(8)).to(torch::kUInt8);
    auto bits = (bit_mult*rand_bin.unflatten(-1,{-1,8})).sum(-1).to(torch::kUInt8);

    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;

        bits = (bit_mult*rand_bin.unflatten(-1,{-1,8}).to(torch::kUInt8)).sum(-1);
        
        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    
    float time_cpu=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time Binary-packbit cpu:\t\t"<<time_cpu*1000.0f/iter<<" msecs."<<std::endl;

    auto output8 = torch::zeros({B,N,D/8}).to(torch::kUInt8);
    //auto output = torch::zeros({1,1,1}).to(torch::kInt64);
    at::native::mps::packbit8Tensor(rand_bin,output8);
    std::cout<<"shapes "<<bits.sizes()<<" , "<<output8.sizes()<<std::endl;
    std::cout<<"types "<<bits.dtype()<<" , "<<output8.dtype()<<std::endl;
    //std::cout<<"diff "<<(bits-output8).abs().sum()<<std::endl;

    //std::cout<<"all close  "<<torch::allclose(bits.to(torch::kUInt8),output8.to(torch::kUInt8))<<std::endl;

    
    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;

        at::native::mps::packbit8Tensor(rand_bin,output8);
        torch::mps::synchronize();

        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    
    float time_mps1=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time Binary-packbit mps:\t\t"<<time_mps1*1000.0f/iter<<" msecs."<<std::endl;
     */
    /*
    //auto rand_bin1 = rand_bin.reshape({1,1,-1});
    //torch::ge(torch::rand({B,128,64}),0.5).to(torch::kInt32);
    auto output = torch::zeros({B,N,D/32}).to(torch::kInt64);
    //auto output = torch::zeros({1,1,1}).to(torch::kInt64);
    at::native::mps::packbitTensor(rand_bin,output);
    std::cout<<"all close  "<<torch::allclose(bits,output)<<std::endl;
    */
    /*
     B = 16;
     N = 2048*4;
     D = 512*2;
     Dv = 64;

    
    auto Q = torch::randn({B,N,D});
    auto K = torch::randn({B,N,D});
    auto V = torch::randn({B,N,Dv});
    //timeval time1,time2;
    auto res1 = torch::zeros({B,N,Dv});*/
/*
    iter = 5;
    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;

        res1 = torch::relu(torch::matmul(Q,K.transpose(-2,-1))).square().matmul(V);
    //auto res1 = res1_;
        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }*/
    /*
    float time_cpu1=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time SDPA cpu:\t\t"<<time_cpu1*1000.0f/iter<<" msecs."<<std::endl;
    iter = 10;
    
    auto Qb = torch::randint(16777216,{B,N,D/32}).to(torch::kInt32);
    auto Kb = torch::randint(16777216,{B,N,D/32}).to(torch::kInt32);
    //auto Qb = torch::randn({B,N,D/32});
    //auto Kb = torch::randn({B,N,D/32});

    auto res3 = torch::zeros({B,N,Dv});
    */
    /*
    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;

        at::native::mps::tiledSDPATensor(Qb,Kb,V,res3);
        torch::mps::synchronize();

        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    */
    /*
    float time_mps2=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time Binary-SDPA mps:\t\t"<<time_mps2*1000.0f/iter<<" msecs."<<std::endl;

    
    auto res2 = torch::zeros({B,N,Dv});
    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;

        at::native::mps::tiledSDPATensorFloat(Q,K,V,res2);
        torch::mps::synchronize();

        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    
    float time_mps=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time SDPA mps(float):\t\t"<<time_mps*1000.0f/iter<<" msecs."<<std::endl;
    std::cout << (res1-res2).norm() << std::endl;

    */
    
    
    /*
    auto A1 = torch::randint(20000,{128,1024,4}).to(torch::kUInt32);
    auto B1 = torch::randint(20000,{128,1024,4}).to(torch::kUInt32);
    auto C1 = torch::zeros({128,1024,1024}).to(torch::kUInt32);
    //auto A1 = torch::pow(2,torch::arange(8)).view({1,8,1}).to(torch::kUInt32);
    //auto B1 = torch::pow(2,torch::arange(8)).view({1,8,1}).to(torch::kUInt32);
    //auto C1 = torch::zeros({1,8,8}).to(torch::kUInt32);
    at::native::mps::hammingTensor(A1,B1,C1);
    //std::cout << C1 << std::endl;
    
    
    auto A2 = torch::randn({128,1024,128});
    auto B2 = torch::randn({128,1024,128});
    auto C2 = torch::zeros({128,1024,1024});
    //auto A1 = torch::pow(2,torch::arange(8)).view({1,8,1}).to(torch::kUInt32);
    //auto B1 = torch::pow(2,torch::arange(8)).view({1,8,1}).to(torch::kUInt32);
    //auto C1 = torch::zeros({1,8,8}).to(torch::kUInt32);
    at::native::mps::matmulTensor(A2,B2,C2);

    
    
    timeval time1,time2;
    int iter = 10;
    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;
        at::native::mps::hammingTensor(A1,B1,C1);
        torch::mps::synchronize();
        //auto maxmem = torch::mps::driver_allocated_memory();
        //std::cout<<maxmem<<std::endl;
        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    
    float time_mps=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time Hamming mps:\t\t"<<time_mps*1000.0f/iter<<" msecs."<<std::endl;

    for(int i=-2;i<iter+1;i++){
        if(i==0){
            gettimeofday(&time1, NULL);
        }
        torch::NoGradGuard no_grad;
        at::native::mps::matmulTensor(A2,B2,C2);

        //auto maxmem1 = torch::mps::driver_allocated_memory();
        //std::cout<<maxmem1<<std::endl;
        if(i==iter-1){
            gettimeofday(&time2, NULL);
        }
    }
    
     time_mps=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    std::cout<<"Time Matmul mps:\t\t"<<time_mps*1000.0f/iter<<" msecs."<<std::endl;

    //auto D = A*B;
    */
    return 0;
}
