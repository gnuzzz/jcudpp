/*
 * JCudpp - Java bindings for CUDPP, the CUDA Data Parallel
 * Primitives Library, to be used with JCuda
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cudpp.h"
#include "cudpp_hash.h"
#include "JCudpp.hpp"
#include "JCudpp_common.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

jfieldID CUDPPConfiguration_algorithm; // CUDPPAlgorithm
jfieldID CUDPPConfiguration_op; // CUDPPOperator
jfieldID CUDPPConfiguration_datatype; // CUDPPDatatype
jfieldID CUDPPConfiguration_options; // unsigned int

jfieldID CUDPPHandle_nativeID; // long

jfieldID CUDPPHashTableConfig_type; // CUDPPHashTableType
jfieldID CUDPPHashTableConfig_kInputSize; // unsigned int
jfieldID CUDPPHashTableConfig_space_usage; // float

/**
 * Called when the library is loaded. Will initialize all
 * required field and method IDs
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }

    Logger::log(LOG_TRACE, "Initializing JCudpp\n");

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;


    // Obtain the fieldIDs of the CUDPPConfiguration class
    if (!init(env, cls, "jcuda/jcudpp/CUDPPConfiguration")) return JNI_ERR;
    if (!init(env, cls, CUDPPConfiguration_algorithm, "algorithm", "I")) return JNI_ERR;
    if (!init(env, cls, CUDPPConfiguration_op,        "op",        "I")) return JNI_ERR;
    if (!init(env, cls, CUDPPConfiguration_datatype,  "datatype",  "I")) return JNI_ERR;
    if (!init(env, cls, CUDPPConfiguration_options,   "options",   "I")) return JNI_ERR;

    // Obtain the fieldIDs of the CUDPPHashTableConfig class
    if (!init(env, cls, "jcuda/jcudpp/CUDPPHashTableConfig")) return JNI_ERR;
    if (!init(env, cls, CUDPPHashTableConfig_type,        "type",        "I")) return JNI_ERR;
    if (!init(env, cls, CUDPPHashTableConfig_kInputSize,  "kInputSize",  "I")) return JNI_ERR;
    if (!init(env, cls, CUDPPHashTableConfig_space_usage, "space_usage", "F")) return JNI_ERR;

    // Obtain the fieldIDs of the CUDPPHandle class
    if (!init(env, cls, "jcuda/jcudpp/CUDPPHandle")) return JNI_ERR;
    if (!init(env, cls, CUDPPHandle_nativeID, "nativeID", "J")) return JNI_ERR;

    return JNI_VERSION_1_4;
}



/*
 * Set the log level
 *
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    setLogLevel
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_jcuda_jcudpp_JCudpp_setLogLevel
  (JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}



//============================================================================



/**
 * Returns the native representation of the given Java object
 */
CUDPPConfiguration getCUDPPConfiguration(JNIEnv *env, jobject config)
{
    CUDPPConfiguration nativeConfig;

    nativeConfig.algorithm = (CUDPPAlgorithm)env->GetIntField(config, CUDPPConfiguration_algorithm);
    nativeConfig.op        = (CUDPPOperator) env->GetIntField(config, CUDPPConfiguration_op);
    nativeConfig.datatype  = (CUDPPDatatype) env->GetIntField(config, CUDPPConfiguration_datatype);
    nativeConfig.options   = (unsigned int)  env->GetIntField(config, CUDPPConfiguration_options);

    return nativeConfig;
}

/**
 * Returns the native representation of the given Java object
 */
CUDPPHashTableConfig getCUDPPHashTableConfig(JNIEnv *env, jobject config)
{
    CUDPPHashTableConfig nativeConfig;

    nativeConfig.type        = (CUDPPHashTableType)env->GetIntField(config,   CUDPPHashTableConfig_type);
    nativeConfig.kInputSize  = (unsigned int)      env->GetIntField(config,   CUDPPHashTableConfig_kInputSize);
    nativeConfig.space_usage = (float)             env->GetFloatField(config, CUDPPHashTableConfig_space_usage);

    return nativeConfig;
}


//============================================================================


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppCreateNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppCreateNative
  (JNIEnv *env, jclass cls, jobject theCudpp)
{
    if (theCudpp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'theCudpp' is null for cudppCreate");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppCreate\n");

    CUDPPHandle nativeTheCudpp;

    int result = cudppCreate(&nativeTheCudpp);

    env->SetLongField(theCudpp, CUDPPHandle_nativeID, (jlong)nativeTheCudpp);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppDestroyNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppDestroyNative
  (JNIEnv *env, jclass cls, jobject theCudpp)
{
    if (theCudpp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'theCudpp' is null for cudppDestroy");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppDestroy\n");

    CUDPPHandle nativeTheCudpp = (CUDPPHandle)env->GetLongField(theCudpp, CUDPPHandle_nativeID);

    int result = cudppDestroy(nativeTheCudpp);

    return result;

}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppPlanNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPConfiguration;JJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppPlanNative
  (JNIEnv *env, jclass cls, jobject cudppHandle, jobject planHandle, jobject config, jlong n, jlong rows, jlong rowPitch)
{
    if (cudppHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cudppHandle' is null for cudppPlan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppPlan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (config == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'config' is null for cudppPlan");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Creating cudppPlan\n");

    CUDPPHandle nativeCudppHandle = (CUDPPHandle)env->GetLongField(cudppHandle, CUDPPHandle_nativeID);
    CUDPPHandle nativePlanHandle;
    CUDPPConfiguration nativeConfig = getCUDPPConfiguration(env, config);

    int result = cudppPlan(nativeCudppHandle, &nativePlanHandle, nativeConfig, (size_t)n, (size_t)rows, (size_t)rowPitch);

    env->SetLongField(planHandle, CUDPPHandle_nativeID, (jlong)nativePlanHandle);

    return result;
}







/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppDestroyPlanNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppDestroyPlanNative
  (JNIEnv *env, jclass cls, jobject planHandle)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppDestroyPlan");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Destroying cudppPlan\n");

    CUDPPHandle nativePlanHandle =  (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    int result = cudppDestroyPlan(nativePlanHandle);
    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppScanNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppScanNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jobject d_in, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppScan");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppScan on %ld elements\n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out = getPointer(env, d_out);
    void *nativeD_in  = getPointer(env, d_in);

    int result = cudppScan(nativePlanHandle, nativeD_out, nativeD_in, (size_t)numElements);

    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppMultiScanNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppMultiScanNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jobject d_in, jlong numElements, jlong numRows)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppMultiScan on %ld elements with %ld rows\n", (long)numElements, (long)numRows);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out = getPointer(env, d_out);
    void *nativeD_in  = getPointer(env, d_in);

    int result = cudppMultiScan(nativePlanHandle, nativeD_out, nativeD_in, (size_t)numElements, (size_t)numRows);

    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppSegmentedScanNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppSegmentedScanNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jobject d_in, jobject d_iflags, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_iflags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_iflags' is null for cudppMultiScan");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppSegmentedScan on %ld elements \n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out = getPointer(env, d_out);
    void *nativeD_in  = getPointer(env, d_in);
    unsigned int *nativeD_iflags  = (unsigned int *)getPointer(env, d_iflags);


    int result = cudppSegmentedScan(nativePlanHandle, nativeD_out, nativeD_in, nativeD_iflags, (size_t)numElements);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppCompactNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppCompactNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jobject d_numValidElements, jobject d_in, jobject d_isValid, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppCompact");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppCompact");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_numValidElements == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_numValidElements' is null for cudppCompact");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppCompact");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_isValid == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_isValid' is null for cudppCompact");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppCompact on %ld elements \n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out                =                getPointer(env, d_out);
    size_t *nativeD_numValidElements = (size_t*)      getPointer(env, d_numValidElements);
    void *nativeD_in                 =                getPointer(env, d_in);
    unsigned int *nativeD_isValid    = (unsigned int*)getPointer(env, d_isValid);

    int result = cudppCompact(nativePlanHandle, nativeD_out, nativeD_numValidElements, nativeD_in, nativeD_isValid, (size_t)numElements);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppReduceNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppReduceNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jobject d_in, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppReduce");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppReduce");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppReduce");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppReduce on %ld elements \n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out                =                getPointer(env, d_out);
    void *nativeD_in                 =                getPointer(env, d_in);

    int result = cudppReduce(nativePlanHandle, nativeD_out, nativeD_in, (size_t)numElements);

    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppRadixSortNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppRadixSortNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_keys, jobject d_values, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppRadixSort");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_keys == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_keys' is null for cudppRadixSort");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppRadixSort on %ld elements\n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_keys = getPointer(env, d_keys);
    void *nativeD_values  = getPointer(env, d_values);

    int result = cudppRadixSort(nativePlanHandle, nativeD_keys, nativeD_values, (size_t)numElements);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppMergeSortNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppMergeSortNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_keys, jobject d_values, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppMergeSort");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_keys == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_keys' is null for cudppMergeSort");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppMergeSort on %ld elements\n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_keys = getPointer(env, d_keys);
    void *nativeD_values  = getPointer(env, d_values);

    int result = cudppMergeSort(nativePlanHandle, nativeD_keys, nativeD_values, (size_t)numElements);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppStringSortNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;BJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppStringSortNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_stringVals, jobject d_address, jbyte termC, jlong numElements, jlong stringArrayLength)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppStringSort");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_stringVals == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_stringVals' is null for cudppStringSort");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_address == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_address' is null for cudppStringSort");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppStringSort on %ld elements\n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    unsigned char *nativeD_stringVals = (unsigned char*)getPointer(env, d_stringVals);
    unsigned int *nativeD_address  = (unsigned int*)getPointer(env, d_address);
	unsigned char nativeTermC = (unsigned char)termC;

	int result = cudppStringSort(nativePlanHandle, nativeD_stringVals, nativeD_address, nativeTermC, (size_t)numElements, (size_t)stringArrayLength);

    return result;

}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppSparseMatrixNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPConfiguration;JJLjcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppSparseMatrixNative
  (JNIEnv *env, jclass cls, jobject cudppHandle, jobject sparseMatrixHandle, jobject config, jlong numNonZeroElements, jlong numRows, jobject A, jobject h_rowIndices, jobject h_indices)
{
    if (cudppHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cudppHandle' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (sparseMatrixHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparseMatrixHandle' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (config == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'config' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (h_rowIndices == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h_rowIndices' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (h_indices == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'h_indices' is null for cudppSparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }


    Logger::log(LOG_TRACE, "Executing cudppSparseMatrix\n");

    CUDPPHandle nativeCudppHandle = (CUDPPHandle)env->GetLongField(cudppHandle, CUDPPHandle_nativeID);
    CUDPPHandle nativeSparseMatrixHandle;
    CUDPPConfiguration nativeConfig = getCUDPPConfiguration(env, config);

    PointerData *APointerData = initPointerData(env, A);
    if (APointerData == NULL)
    {
        return JCUDPP_INTERNAL_ERROR;
    }
    PointerData *h_rowIndicesPointerData = initPointerData(env, h_rowIndices);
    if (h_rowIndicesPointerData == NULL)
    {
        return JCUDPP_INTERNAL_ERROR;
    }
    PointerData *h_indicesPointerData = initPointerData(env, h_indices);
    if (h_indicesPointerData == NULL)
    {
        return JCUDPP_INTERNAL_ERROR;
    }

    int result = cudppSparseMatrix(
        nativeCudppHandle,
        &nativeSparseMatrixHandle,
        nativeConfig,
        (size_t)numNonZeroElements,
        (size_t)numRows,
        (void*)APointerData->getPointer(env),
        (unsigned int*)h_rowIndicesPointerData->getPointer(env),
        (unsigned int*)h_indicesPointerData->getPointer(env));

    if (!releasePointerData(env, APointerData)) return JCUDPP_INTERNAL_ERROR;
    if (!releasePointerData(env, h_rowIndicesPointerData)) return JCUDPP_INTERNAL_ERROR;
    if (!releasePointerData(env, h_indicesPointerData)) return JCUDPP_INTERNAL_ERROR;

    env->SetLongField(sparseMatrixHandle, CUDPPHandle_nativeID, (jlong)nativeSparseMatrixHandle);

    return result;
}







/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppDestroySparseMatrixNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppDestroySparseMatrixNative
  (JNIEnv *env, jclass cls, jobject sparseMatrixHandle)
{
    if (sparseMatrixHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparseMatrixHandle' is null for cudppDestroySparseMatrix");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppDestroySparseMatrix\n");

    CUDPPHandle nativeSparseMatrixHandle = (CUDPPHandle)env->GetLongField(sparseMatrixHandle, CUDPPHandle_nativeID);

    int result = cudppDestroySparseMatrix(nativeSparseMatrixHandle);
    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppSparseMatrixVectorMultiplyNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppSparseMatrixVectorMultiplyNative
  (JNIEnv *env, jclass cls, jobject sparseMatrixHandle, jobject d_y, jobject d_x)
{
    if (sparseMatrixHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparseMatrixHandle' is null for cudppSparseMatrixVectorMultiply");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_y' is null for cudppSparseMatrixVectorMultiply");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_x' is null for cudppSparseMatrixVectorMultiply");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppSparseMatrixVectorMultiply\n");


    CUDPPHandle nativeSparseMatrixHandle = (CUDPPHandle)env->GetLongField(sparseMatrixHandle, CUDPPHandle_nativeID);
    void *nativeD_y = getPointer(env, d_y);
    void *nativeD_x = getPointer(env, d_x);

    int result = cudppSparseMatrixVectorMultiply(nativeSparseMatrixHandle, nativeD_y, nativeD_x);

    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppRandNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppRandNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_out, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppRand");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppRand");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppRand on %ld elements \n", (long)numElements);

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_out = getPointer(env, d_out);

    int result = cudppRand(nativePlanHandle, nativeD_out, (size_t)numElements);

    return result;

}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppRandSeedNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppRandSeedNative
  (JNIEnv *env, jclass cls, jobject planHandle, jint seed)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppRandSeed");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppRandSeed\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);

    int result = cudppRandSeed(nativePlanHandle, (unsigned int)seed);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppTridiagonalNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppTridiagonalNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject a, jobject b, jobject c, jobject d, jobject x, jint systemSize, jint numSystems)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (a == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'a' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (b == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'b' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (c == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'c' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cudppTridiagonal");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppTridiagonal\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeA = getPointer(env, a);
    void *nativeB = getPointer(env, b);
    void *nativeC = getPointer(env, c);
    void *nativeD = getPointer(env, d);
    void *nativeX = getPointer(env, x);

    int result = cudppTridiagonal(nativePlanHandle, nativeA, nativeB, nativeC, nativeD, nativeX, (int)systemSize, (int)numSystems);

    return result;

}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppCompressNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppCompressNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_uncompressed, jobject d_bwtIndex, jobject d_histSize, jobject d_hist, jobject d_encodeOffset, jobject d_compressedSize, jobject d_compressed, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_uncompressed == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_uncompressed' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_bwtIndex == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_bwtIndex' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
	/* May be null
    if (d_histSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_histSize' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
	*/
    if (d_hist == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_hist' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_encodeOffset == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_encodeOffset' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_compressedSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_compressedSize' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_compressed == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_compressed' is null for cudppCompress");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppCompress\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    unsigned char *nativeD_uncompressed = (unsigned char*)getPointer(env, d_uncompressed);
    int *nativeD_bwtIndex = (int*)getPointer(env, d_bwtIndex);
    unsigned int *nativeD_histSize = (unsigned int*)getPointer(env, d_histSize);
    unsigned int *nativeD_hist = (unsigned int*)getPointer(env, d_hist);
    unsigned int *nativeD_encodeOffset = (unsigned int*)getPointer(env, d_encodeOffset);
	unsigned int *nativeD_compressedSize = (unsigned int*)getPointer(env, d_compressedSize);
    unsigned int *nativeD_compressed = (unsigned int*)getPointer(env, d_compressed);
	size_t nativeNumElements = (size_t)numElements;

	int result = cudppCompress(nativePlanHandle, nativeD_uncompressed, nativeD_bwtIndex, nativeD_histSize, nativeD_hist, nativeD_encodeOffset, nativeD_compressedSize, nativeD_compressed, nativeNumElements);

    return result;


}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppBurrowsWheelerTransformNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppBurrowsWheelerTransformNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_in, jobject d_out, jobject d_index, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppBurrowsWheelerTransform");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppBurrowsWheelerTransform");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppBurrowsWheelerTransform");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_index == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_index' is null for cudppBurrowsWheelerTransform");
        return JCUDPP_INTERNAL_ERROR;
    }


    Logger::log(LOG_TRACE, "Executing cudppBurrowsWheelerTransform\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    unsigned char *nativeD_in = (unsigned char*)getPointer(env, d_in);
    unsigned char *nativeD_out = (unsigned char*)getPointer(env, d_out);
    int *nativeD_index = (int*)getPointer(env, d_index);
	size_t nativeNumElements = (size_t)numElements;

	int result = cudppBurrowsWheelerTransform(nativePlanHandle, nativeD_in, nativeD_out, nativeD_index, nativeNumElements);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppMoveToFrontTransformNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppMoveToFrontTransformNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_in, jobject d_out, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppMoveToFrontTransform");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_in == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_in' is null for cudppMoveToFrontTransform");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_out' is null for cudppMoveToFrontTransform");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppMoveToFrontTransform\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    unsigned char *nativeD_in = (unsigned char*)getPointer(env, d_in);
    unsigned char *nativeD_out = (unsigned char*)getPointer(env, d_out);
	size_t nativeNumElements = (size_t)numElements;

	int result = cudppMoveToFrontTransform(nativePlanHandle, nativeD_in, nativeD_out, nativeNumElements);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppListRankNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;Ljcuda/Pointer;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppListRankNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_ranked_values, jobject d_unranked_values, jobject d_next_indices, jlong head, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppListRank");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_ranked_values == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_ranked_values' is null for cudppListRank");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_unranked_values == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_unranked_values' is null for cudppListRank");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_next_indices == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_next_indices' is null for cudppListRank");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppListRank\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    void *nativeD_ranked_values = getPointer(env, d_ranked_values);
	void *nativeD_unranked_values = getPointer(env, d_unranked_values);
	void *nativeD_next_indices = getPointer(env, d_next_indices);
	size_t nativeHead = (size_t)head;
	size_t nativeNumElements = (size_t)numElements;

	int result = cudppListRank(nativePlanHandle, nativeD_ranked_values, nativeD_unranked_values, nativeD_next_indices, nativeHead, nativeNumElements);

    return result;

}


/*
* Class:     jcuda_jcudpp_JCudpp
* Method:    cudppSuffixArrayNative
* Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppSuffixArrayNative
  (JNIEnv *env, jclass cls, jobject planHandle, jobject d_str, jobject d_keys_sa, jlong numElements)
{
    if (planHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'planHandle' is null for cudppSuffixArray");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_str == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_str' is null for cudppSuffixArray");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_keys_sa == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_keys_sa' is null for cudppSuffixArray");
        return JCUDPP_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudppSuffixArray\n");

    CUDPPHandle nativePlanHandle = (CUDPPHandle)env->GetLongField(planHandle, CUDPPHandle_nativeID);
    unsigned char *nativeD_str = (unsigned char*)getPointer(env, d_str);
    unsigned int *nativeD_keys_sa = (unsigned int*)getPointer(env, d_keys_sa);
    size_t nativeNumElements = (size_t)numElements;

    int result = cudppSuffixArray(nativePlanHandle, nativeD_str, nativeD_keys_sa, nativeNumElements);

    return result;
}



/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppHashTableNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPHashTableConfig;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppHashTableNative
  (JNIEnv *env, jclass cls, jobject cudppHandle, jobject plan, jobject config)
{
    if (cudppHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cudppHandle' is null for cudppHashTable");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppHashTable");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (config == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'config' is null for cudppHashTable");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppHashTable\n");

    CUDPPHandle nativeCudppHandle = (CUDPPHandle)env->GetLongField(cudppHandle, CUDPPHandle_nativeID);
    CUDPPHandle nativePlan;
    CUDPPHashTableConfig nativeConfig = getCUDPPHashTableConfig(env, config);

    int result = cudppHashTable(nativeCudppHandle, &nativePlan, &nativeConfig);

    env->SetLongField(plan, CUDPPHandle_nativeID, (jlong)nativePlan);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppDestroyHashTableNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/jcudpp/CUDPPHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppDestroyHashTableNative
  (JNIEnv *env, jclass cls, jobject cudppHandle, jobject plan)
{
    if (cudppHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cudppHandle' is null for cudppDestroyHashTable");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppDestroyHashTable");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppDestroyHashTable\n");

    CUDPPHandle nativeCudppHandle = (CUDPPHandle)env->GetLongField(cudppHandle, CUDPPHandle_nativeID);
    CUDPPHandle nativePlan = (CUDPPHandle)env->GetLongField(plan, CUDPPHandle_nativeID);

    int result = cudppDestroyHashTable(nativeCudppHandle, nativePlan);

    return result;
}


/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppHashInsertNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppHashInsertNative
  (JNIEnv *env, jclass cls, jobject plan, jobject d_keys, jobject d_vals, jlong num)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppHashInsert");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_keys == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_keys' is null for cudppHashInsert");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_vals == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_vals' is null for cudppHashInsert");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppHashInsert\n");

    CUDPPHandle nativePlan = (CUDPPHandle)env->GetLongField(plan, CUDPPHandle_nativeID);
    void *nativeD_keys = getPointer(env, d_keys);
    void *nativeD_vals = getPointer(env, d_vals);

    int result = cudppHashInsert(nativePlan, nativeD_keys, nativeD_vals, (size_t)num);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppHashRetrieveNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppHashRetrieveNative
  (JNIEnv *env, jclass cls, jobject plan, jobject d_keys, jobject d_vals, jlong num)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppHashRetrieve");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_keys == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_keys' is null for cudppHashRetrieve");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_vals == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_vals' is null for cudppHashRetrieve");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppHashRetrieve\n");

    CUDPPHandle nativePlan = (CUDPPHandle)env->GetLongField(plan, CUDPPHandle_nativeID);
    void *nativeD_keys = getPointer(env, d_keys);
    void *nativeD_vals = getPointer(env, d_vals);

    int result = cudppHashRetrieve(nativePlan, nativeD_keys, nativeD_vals, (size_t)num);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppMultivalueHashGetValuesSizeNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppMultivalueHashGetValuesSizeNative
  (JNIEnv *env, jclass cls, jobject plan, jintArray size)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppMultivalueHashGetValuesSize");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (size == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'size' is null for cudppMultivalueHashGetValuesSize");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppMultivalueHashGetValuesSize\n");

    CUDPPHandle nativePlan = (CUDPPHandle)env->GetLongField(plan, CUDPPHandle_nativeID);
    unsigned int nativeSize;

    int result = cudppMultivalueHashGetValuesSize(nativePlan, &nativeSize);

    set(env, size, 0, nativeSize);

    return result;
}

/*
 * Class:     jcuda_jcudpp_JCudpp
 * Method:    cudppMultivalueHashGetAllValuesNative
 * Signature: (Ljcuda/jcudpp/CUDPPHandle;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcudpp_JCudpp_cudppMultivalueHashGetAllValuesNative
  (JNIEnv *env, jclass cls, jobject plan, jobject d_vals)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cudppMultivalueHashGetAllValues");
        return JCUDPP_INTERNAL_ERROR;
    }
    if (d_vals == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'd_vals' is null for cudppMultivalueHashGetAllValues");
        return JCUDPP_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudppMultivalueHashGetAllValues\n");

    CUDPPHandle nativePlan = (CUDPPHandle)env->GetLongField(plan, CUDPPHandle_nativeID);
    unsigned int *nativeD_vals = NULL;

    int result = cudppMultivalueHashGetAllValues(nativePlan, &nativeD_vals);

    setPointer(env, d_vals, (jlong)nativeD_vals);

    return result;
}
