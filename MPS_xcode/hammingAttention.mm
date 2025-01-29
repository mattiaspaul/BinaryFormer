#import "hammingAttention.h"
#import <Foundation/Foundation.h>
#include <ATen/core/Tensor.h>
#include <math.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>


namespace at::native {
namespace mps {

/*- (MPSGraphTensor *) HammingDistanceWithPrimaryTensor:(MPSGraphTensor *) primaryTensor
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
 
 - (MPSGraphTensor *) bitwisePopulationCountWithTensor:(MPSGraphTensor *) tensor
                                                  name:(NSString *) name;
 - (MPSGraphTensor *) bitwiseXORWithPrimaryTensor:(MPSGraphTensor *) primaryTensor
                                  secondaryTensor:(MPSGraphTensor *) secondaryTensor
                                             name:(NSString *) name;
 */
void matmulTensor(const at::Tensor& query,const Tensor& key,const Tensor& output){
    //N = sequence length
    const int64_t B = query.size(0);
    const int64_t N = query.size(1);
    const int64_t C = query.size(2);

    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        

        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*C*N] shape:@[@(B),@(N),@(C)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*C*N] shape:@[@(B),@(C),@(N)] dataType:MPSDataTypeFloat32 name:nil];
        //MPSGraphTensor* difference1 = [mpsGraph bitwiseXORWithPrimaryTensor:key1 secondaryTensor:query1 name:nil];
        //MPSGraphTensor* output1 = [mpsGraph bitwisePopulationCountWithTensor:difference1  name:nil];
        MPSGraphTensor *output1 = [mpsGraph matrixMultiplicationWithPrimaryTensor:query1 secondaryTensor: key1 name: nil];

        
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[output1] targetOperations:@[]];
        [result[output1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
    }//end autorelease
}


void hammingTensor(const at::Tensor& query,const Tensor& key,const Tensor& output){
    const int64_t B = query.size(0);
    const int64_t N = query.size(1);
    const int64_t C = query.size(2);
    //NSLog(@"Hello, hamming Na,M,K %lld, %lld, %lld!",B,N,C);
    const int64_t B1 = output.size(0);
    const int64_t N1 = output.size(1);
    const int64_t N2 = output.size(2);

    //NSLog(@"output will be Na,M,N %lld, %lld, %lld!",B,N,N);
    //NSLog(@"provided memory tensor is Na,M,N %lld, %lld, %lld!",B1,N1,N2);
    
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        

        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*C*N] shape:@[@(B),@(N),@(C)] dataType:MPSDataTypeUInt32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*C*N] shape:@[@(B),@(N),@(C)] dataType:MPSDataTypeUInt32 name:nil];
        //MPSGraphTensor* difference1 = [mpsGraph bitwiseXORWithPrimaryTensor:key1 secondaryTensor:query1 name:nil];
        //MPSGraphTensor* output1 = [mpsGraph bitwisePopulationCountWithTensor:difference1  name:nil];
        MPSGraphTensor *output1 = [mpsGraph HammingDistanceWithPrimaryTensor:query1 secondaryTensor: key1 resultDataType:MPSDataTypeUInt32 name: nil];

        
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[output1] targetOperations:@[]];
        [result[output1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
    }//end autorelease
}

void packbitTensor(const Tensor& input,const Tensor& output){
    
    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t D = input.size(2)/32;
    const int64_t D32 = 32;
    uint32_t numbers[32];
    for(uint32_t i=0;i<32;i++){
        numbers[i] = i;
    }
    //NSLog(@"shape input %@",input.shape);

    
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *input1 = [mpsGraph variableWithData:[NSData dataWithBytes:input.data_ptr() length:B*N*D*D32] shape:@[@(B),@(N),@(D),@(D32)] dataType:MPSDataTypeUInt8 name:nil];

        MPSGraphTensor *input32 = [mpsGraph castTensor:input1 toType:MPSDataTypeUInt32 name:nil];
        MPSGraphTensor *arange32 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:4*32] shape:@[@1,@1,@1,@32] dataType:MPSDataTypeUInt32 name:nil];
        MPSGraphTensor *two32 = [mpsGraph constantWithScalar:2 shape:@[@1,@1,@1,@32]  dataType:MPSDataTypeUInt32];
        MPSGraphTensor *bit_factor32 = [mpsGraph powerWithPrimaryTensor:two32 secondaryTensor:arange32 name:nil];

        MPSGraphTensor *bit_mult = [mpsGraph multiplicationWithPrimaryTensor:bit_factor32 secondaryTensor:input32 name:nil];

        MPSGraphTensor *result1 = [mpsGraph reductionSumWithTensor:bit_mult axis:-1 name:nil];

        MPSGraphTensor *result64 = [mpsGraph castTensor:result1 toType:MPSDataTypeInt64 name:nil];


        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[result64] targetOperations:@[]];
        [result[result64].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}

static MPSGraphTensor* packbit8(MPSGraph* mpsGraph, MPSGraphTensor* input1,int64_t B,int64_t N,int64_t D1){
    int64_t D = D1/8;
    int64_t D8 = 8;
    uint8_t numbers[8];
    for(uint8_t i=0;i<8;i++){
        numbers[i] = powl(2,i);
    }
    MPSGraphTensor *bit_factor8 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:8] shape:@[@1,@1,@1,@8] dataType:MPSDataTypeUInt8 name:nil];
    MPSGraphTensor *bit_mult = [mpsGraph multiplicationWithPrimaryTensor:bit_factor8 secondaryTensor:input1 name:nil];
    return [mpsGraph reductionSumWithTensor:bit_mult axis:-1 name:nil];
}

static MPSGraphTensor* packbit32(MPSGraph* mpsGraph, MPSGraphTensor* input1,int64_t B,int64_t N,int64_t D1){
    int64_t D = D1/32;
    int64_t D8 = 32;
    uint32_t numbers[32];
    for(uint32_t i=0;i<32;i++){
        numbers[i] = powl(2,i);
    }
    MPSGraphTensor *bit_factor8 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:32] shape:@[@1,@1,@1,@32] dataType:MPSDataTypeUInt32 name:nil];
    MPSGraphTensor *bit_mult = [mpsGraph multiplicationWithPrimaryTensor:bit_factor8 secondaryTensor:input1 name:nil];
    return [mpsGraph reductionSumWithTensor:bit_mult axis:-1 name:nil];
}



void packbit8Tensor(const Tensor& input,const Tensor& output){
    
    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t D = input.size(2)/8;
    const int64_t D8 = 8;
    uint8_t numbers[8];
    for(uint8_t i=0;i<8;i++){
        numbers[i] = powl(2,i);
    }
    //NSLog(@"shape input %d, %d, %d, %d",B,N,D,D8);

    
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *input8 = [mpsGraph variableWithData:[NSData dataWithBytes:input.data_ptr() length:B*N*D*D8] shape:@[@(B),@(N),@(D),@(D8)] dataType:MPSDataTypeUInt8 name:nil];
//        NSLog(@"shape input8 %@",input8.shape);

        //MPSGraphTensor *input32 = [mpsGraph castTensor:input1 toType:MPSDataTypeUInt32 name:nil];
        //MPSGraphTensor *arange8 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:8] shape:@[@1,@1,@1,@8] dataType:MPSDataTypeUInt8 name:nil];
        //MPSGraphTensor *two8 = [mpsGraph constantWithScalar:2 shape:@[@1,@1,@1,@8]  dataType:MPSDataTypeUInt8];
        //MPSGraphTensor *bit_factor8 = [mpsGraph powerWithPrimaryTensor:two8 secondaryTensor:arange8 name:nil];
      //  NSLog(@"shape bit_factor8 %@",bit_factor8.shape);
        MPSGraphTensor *bit_factor8 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:8] shape:@[@1,@1,@1,@8] dataType:MPSDataTypeUInt8 name:nil];

        MPSGraphTensor *bit_mult = [mpsGraph multiplicationWithPrimaryTensor:bit_factor8 secondaryTensor:input8 name:nil];
        //NSLog(@"shape bit_mult %@",bit_mult.shape);

        MPSGraphTensor *result1 = [mpsGraph reductionSumWithTensor:bit_mult axis:-1 name:nil];
        //NSLog(@"shape result1 %@",result1.shape);



        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[result1] targetOperations:@[]];
        [result[result1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}



void tiledHammingAttentionReLU(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& output){
    
    const int64_t B = query.size(0);
    const int64_t N = query.size(1);
    const int64_t D1 = query.size(2);
    const int64_t D = D1/32;

    const int64_t Dv = value.size(2);
    const int64_t tiles = 1;
    const int64_t blk = N/tiles;
    
    uint32_t numbers[32];
    for(uint32_t i=0;i<32;i++){
        numbers[i] = powl(2,i);
    }

    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *query1f = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D),@32] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1f = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D),@32] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *value1 = [mpsGraph variableWithData:[NSData dataWithBytes:value.data_ptr() length:4*B*N*Dv] shape:@[@(B),@(N),@(Dv)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N),@(Dv)]  dataType:MPSDataTypeFloat32];
        
        MPSGraphTensor *thresh1 = [mpsGraph constantWithScalar:0.0f shape:@[@1,@1,@1,@1]  dataType:MPSDataTypeFloat32];

        MPSGraphTensor *query1f_ = [mpsGraph greaterThanWithPrimaryTensor:query1f secondaryTensor:thresh1 name:nil];
        MPSGraphTensor *query1i = [mpsGraph castTensor:query1f_ toType:MPSDataTypeUInt32 name:nil];
        MPSGraphTensor *bit_factor8 = [mpsGraph variableWithData:[NSData dataWithBytes:numbers length:32] shape:@[@1,@1,@1,@32] dataType:MPSDataTypeUInt32 name:nil];
        MPSGraphTensor *bit_mult = [mpsGraph multiplicationWithPrimaryTensor:bit_factor8 secondaryTensor:query1i name:nil];
        
        MPSGraphTensor *query1_ =  [mpsGraph reductionSumWithTensor:bit_mult axis:-1 name:nil];
        MPSGraphTensor *query1 =  [mpsGraph squeezeTensor:query1_ axis:-1 name:nil];
        
        MPSGraphTensor *key1f_ = [mpsGraph greaterThanWithPrimaryTensor:key1f secondaryTensor:thresh1 name:nil];
        MPSGraphTensor *key1i = [mpsGraph castTensor:key1f_ toType:MPSDataTypeUInt32 name:nil];
        
        MPSGraphTensor *bit_multk = [mpsGraph multiplicationWithPrimaryTensor:bit_factor8 secondaryTensor:key1i name:nil];
        MPSGraphTensor *key1_ =  [mpsGraph reductionSumWithTensor:bit_multk axis:-1 name:nil];
        MPSGraphTensor *key1 = [mpsGraph squeezeTensor:key1_ axis:-1 name:nil];

    
        NSLog(@"output of packbit32 %@ %@",query1.shape,key1.shape);



        for(int64_t iter=0;iter<10;iter++){
            res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N),@(Dv)]  dataType:MPSDataTypeFloat32];
            
            for(int64_t i=0;i<tiles;i++){
                for(int64_t j=0;j<tiles;j++){
                    
                    MPSGraphTensor *q_i = [mpsGraph sliceTensor:query1  dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *k_j = [mpsGraph sliceTensor:key1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *v_j = [mpsGraph sliceTensor:value1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *k_t = [mpsGraph transposeTensor:k_j dimension:-2 withDimension:-1 name:nil];
                    
                    MPSGraphTensor *out_int = [mpsGraph HammingDistanceWithPrimaryTensor:q_i secondaryTensor: k_j resultDataType:MPSDataTypeUInt32 name: nil];
                    MPSGraphTensor *out_i1 = [mpsGraph castTensor:out_int toType:MPSDataTypeFloat32 name:nil];
                    MPSShape *shape1 = out_i1.shape;
                    NSLog(@"output of sliced Hamming distance %@",shape1);
                    
                    //MPSGraphTensor *out_relu = [mpsGraph reLUWithTensor:out_i name:nil];
                    MPSGraphTensor *out_relu_sq = [mpsGraph squareWithTensor:out_i1 name:nil];
                    MPSGraphTensor *out_mm_v = [mpsGraph matrixMultiplicationWithPrimaryTensor:out_relu_sq secondaryTensor:v_j name:nil];
                    //to do plus add
                    MPSGraphTensor *slice_add = [mpsGraph sliceTensor:res1 dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *plus_add = [mpsGraph additionWithPrimaryTensor:out_mm_v secondaryTensor:slice_add name:nil];
                    NSLog(@"output of sliced plus_add%@",plus_add.shape);

                    res1 = [mpsGraph sliceUpdateDataTensor:res1 updateTensor:plus_add starts:@[@0,@(i*blk),@0] ends:@[@(B),@((i+1)*blk),@(Dv)] strides:@[@1,@1,@1] name:nil];
                }
            }
        }
        
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[res1] targetOperations:@[]];
        [result[res1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}
void tiledSDPAfloatReLU(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& output){
    
    const int64_t B = query.size(0);
    const int64_t N = query.size(1);
    const int64_t D = query.size(2);
    
    const int64_t Dv = value.size(2);
    const int64_t tiles = 1;
    const int64_t blk = N/tiles;
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *value1 = [mpsGraph variableWithData:[NSData dataWithBytes:value.data_ptr() length:4*B*N*Dv] shape:@[@(B),@(N),@(Dv)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N),@(Dv)]  dataType:MPSDataTypeFloat32];
        //NSLog(@"running 10 iterations");
        for(int64_t iter=0;iter<10;iter++){
            res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N),@(Dv)]  dataType:MPSDataTypeFloat32];
            for(int64_t i=0;i<tiles;i++){
                for(int64_t j=0;j<tiles;j++){
                    
                    MPSGraphTensor *q_i = [mpsGraph sliceTensor:query1  dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *k_j = [mpsGraph sliceTensor:key1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *v_j = [mpsGraph sliceTensor:value1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *k_t = [mpsGraph transposeTensor:k_j dimension:-2 withDimension:-1 name:nil];
                    MPSGraphTensor *out_i = [mpsGraph matrixMultiplicationWithPrimaryTensor:q_i secondaryTensor:k_t name:nil];
                    //MPSShape *shape1 = out_i.shape;
                    //NSLog(@"output of sliced Hamming distance %@",shape1);
                    
                    MPSGraphTensor *out_relu = [mpsGraph reLUWithTensor:out_i name:nil];
                    MPSGraphTensor *out_relu_sq = [mpsGraph squareWithTensor:out_relu name:nil];
                    MPSGraphTensor *out_mm_v = [mpsGraph matrixMultiplicationWithPrimaryTensor:out_relu_sq secondaryTensor:v_j name:nil];
                    //to do plus add
                    MPSGraphTensor *slice_add = [mpsGraph sliceTensor:res1 dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *plus_add = [mpsGraph additionWithPrimaryTensor:out_mm_v secondaryTensor:slice_add name:nil];
                    res1 = [mpsGraph sliceUpdateDataTensor:res1 updateTensor:plus_add starts:@[@0,@(i*blk),@0] ends:@[@(B),@((i+1)*blk),@(Dv)] strides:@[@1,@1,@1] name:nil];
                }
            }
        }
        
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[res1] targetOperations:@[]];
        [result[res1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}

void tiledSDPAfloat(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& output){
    
    const int64_t B = query.size(0);
    const int64_t N = query.size(1);
    const int64_t D = query.size(2);
    
    const int64_t Dv = value.size(2);
    const int64_t tiles = 1;
    const int64_t blk = N/tiles;
    float scale = 1.0f/sqrtf(float(D));
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*N*D] shape:@[@(B),@(N),@(D)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *value1 = [mpsGraph variableWithData:[NSData dataWithBytes:value.data_ptr() length:4*B*N*Dv] shape:@[@(B),@(N),@(Dv)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N),@(Dv)]  dataType:MPSDataTypeFloat32];
        MPSGraphTensor *max1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N)]  dataType:MPSDataTypeFloat32];
        MPSGraphTensor *lin1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(N)]  dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scale1 = [mpsGraph constantWithScalar:scale shape:@[@1,@1,@1]  dataType:MPSDataTypeFloat32];
        //NSLog(@"running 10 iterations");
        for(int64_t iter=0;iter<10;iter++){
            
            for(int64_t i=0;i<tiles;i++){
                for(int64_t j=0;j<tiles;j++){
                    MPSGraphTensor *q_i = [mpsGraph sliceTensor:query1  dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *O_i = [mpsGraph sliceTensor:res1  dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *k_j = [mpsGraph sliceTensor:key1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *v_j = [mpsGraph sliceTensor:value1  dimension:1 start:j*blk length:blk name:nil];
                    MPSGraphTensor *k_t = [mpsGraph transposeTensor:k_j dimension:-2 withDimension:-1 name:nil];
                    MPSGraphTensor *m_i = [mpsGraph sliceTensor:max1 dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *l_i = [mpsGraph sliceTensor:lin1 dimension:1 start:i*blk length:blk name:nil];
                    MPSGraphTensor *Sij_ = [mpsGraph matrixMultiplicationWithPrimaryTensor:q_i secondaryTensor:k_t name:nil];
                    MPSGraphTensor *Sij = [mpsGraph multiplicationWithPrimaryTensor:Sij_ secondaryTensor:scale1 name:nil];
                    MPSGraphTensor *Mij_ = [mpsGraph reductionMaximumWithTensor:Sij axis:-1 name:nil];
                    MPSGraphTensor *Mij = [mpsGraph squeezeTensor:Mij_ axis:-1 name:nil];
                    MPSGraphTensor *Mij__ = [mpsGraph subtractionWithPrimaryTensor:Sij secondaryTensor:Mij_ name:nil];
                    MPSGraphTensor *Pij = [mpsGraph exponentWithTensor:Mij__ name:nil];
                    MPSGraphTensor *out_ij = [mpsGraph matrixMultiplicationWithPrimaryTensor:Pij secondaryTensor:v_j name:nil];
                    MPSGraphTensor *Lij_ = [mpsGraph reductionSumWithTensor:Pij axis:-1 name:nil];
                    MPSGraphTensor *Lij = [mpsGraph squeezeTensor:Lij_ axis:-1 name:nil];
                    MPSGraphTensor *M_new = [mpsGraph maximumWithPrimaryTensor:m_i secondaryTensor:Mij name:nil];
                    MPSGraphTensor *L_new1__ = [mpsGraph subtractionWithPrimaryTensor:m_i secondaryTensor:M_new name:nil];
                    MPSGraphTensor *L_new1_ = [mpsGraph exponentWithTensor:L_new1__ name:nil];
                    MPSGraphTensor *L_new1 = [mpsGraph multiplicationWithPrimaryTensor:L_new1_ secondaryTensor:l_i name:nil];
                    MPSGraphTensor *L_new2__ = [mpsGraph subtractionWithPrimaryTensor:Mij secondaryTensor:M_new name:nil];
                    MPSGraphTensor *L_new2_ = [mpsGraph exponentWithTensor:L_new2__ name:nil];
                    MPSGraphTensor *L_new2 = [mpsGraph multiplicationWithPrimaryTensor:L_new2_ secondaryTensor:Lij name:nil];
                    MPSGraphTensor *L_new = [mpsGraph additionWithPrimaryTensor:L_new1 secondaryTensor:L_new2 name:nil];
                    MPSGraphTensor *L_new_ = [mpsGraph expandDimsOfTensor:L_new axis:-1 name:nil];
                    MPSGraphTensor *l_i1 = [mpsGraph expandDimsOfTensor:l_i axis:-1 name:nil];
                    MPSGraphTensor *temp1__ =  [mpsGraph expandDimsOfTensor:L_new1_ axis:-1 name:nil];//exp(Mi - M_new)
                    MPSGraphTensor *temp1_ =  [mpsGraph multiplicationWithPrimaryTensor:temp1__ secondaryTensor:l_i1 name:nil];
                    MPSGraphTensor *temp1 =  [mpsGraph multiplicationWithPrimaryTensor:temp1_ secondaryTensor:O_i name:nil];
                    MPSGraphTensor *temp2_ =  [mpsGraph expandDimsOfTensor:L_new2_ axis:-1 name:nil];//exp(Mij - M_new)
                    MPSGraphTensor *temp2 =  [mpsGraph multiplicationWithPrimaryTensor:temp2_ secondaryTensor:out_ij name:nil];
                    MPSGraphTensor *temp_ = [mpsGraph additionWithPrimaryTensor:temp1 secondaryTensor:temp2 name:nil];
                    MPSGraphTensor *temp = [mpsGraph divisionWithPrimaryTensor:temp_ secondaryTensor:L_new_ name:nil];
                    res1 = [mpsGraph sliceUpdateDataTensor:res1 updateTensor:temp starts:@[@0,@(i*blk),@0] ends:@[@(B),@((i+1)*blk),@(Dv)] strides:@[@1,@1,@1] name:nil];
                    max1 = [mpsGraph sliceUpdateDataTensor:max1 updateTensor:M_new starts:@[@0,@(i*blk)] ends:@[@(B),@((i+1)*blk)] strides:@[@1,@1] name:nil];
                    lin1 = [mpsGraph sliceUpdateDataTensor:lin1 updateTensor:L_new starts:@[@0,@(i*blk)] ends:@[@(B),@((i+1)*blk)] strides:@[@1,@1] name:nil];
                }
            }
        }
        
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[res1] targetOperations:@[]];
        [result[res1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}




void multiplyTensor(const Tensor& query,const Tensor& key,const Tensor& output){
    const int64_t B = query.size(0);
    const int64_t C = query.size(1);
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*C] shape:@[@(B),@(C)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*C] shape:@[@(B),@(C)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *res1 = [mpsGraph constantWithScalar:0.0f shape:@[@(B),@(C)]  dataType:MPSDataTypeFloat32];

        for(int i=0;i<8;i++){
            MPSGraphTensor *q_i = [mpsGraph sliceTensor:query1  dimension:0 start:i*8 length:8 name:nil];
            MPSGraphTensor *k_i = [mpsGraph sliceTensor:key1  dimension:0 start:i*8 length:8 name:nil];
            MPSGraphTensor *out_i = [mpsGraph multiplicationWithPrimaryTensor:q_i secondaryTensor:k_i name:nil];
            res1 = [mpsGraph sliceUpdateDataTensor:res1 updateTensor:out_i starts:@[@(i*8),@0] ends:@[@((i+1)*8),@64] strides:@[@1,@1] name:nil];
        }
        
        //MPSGraphTensor* output1 = [mpsGraph multiplicationWithPrimaryTensor:key1 secondaryTensor:query1 name:nil];
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[res1] targetOperations:@[]];
        [result[res1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}

} //end of namespaces
}

/*const Tensor& query,const Tensor& key,const Tensor& output
    const int64_t B = query.size(0);
    const int64_t C = query.size(1);
    const int64_t N = query.size(2);
    //N = sequence length
    @autoreleasepool {
        NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
        id<MTLDevice> device = availableDevices[0];
        id<MTLCommandQueue> _Nonnull gCommandQueue;
        
        gCommandQueue = [device newCommandQueue];
        MPSGraph *mpsGraph = [MPSGraph new];
        
        MPSGraphTensor *query1 = [mpsGraph variableWithData:[NSData dataWithBytes:query.data_ptr() length:4*B*C*N] shape:@[@(B),@(C),@(N)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *key1 = [mpsGraph variableWithData:[NSData dataWithBytes:key.data_ptr() length:4*B*C*N] shape:@[@(B),@(C),@(N)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* output1 = [mpsGraph multiplicationWithPrimaryTensor:key1 secondaryTensor:query1 name:nil];
        MPSGraphTensorDataDictionary *result = [mpsGraph runWithMTLCommandQueue: gCommandQueue feeds:@{} targetTensors:@[output1] targetOperations:@[]];
        [result[output1].mpsndarray readBytes:output.data_ptr() strideBytes:Nil];
        
        
    }
}


}
}*/
