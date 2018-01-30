/*
 * JCudpp - Java bindings for CUDPP, the CUDA Data Parallel
 * Primitives Library, to be used with JCuda
 *
 * Copyright (c) 2009-2012 Marco Hutter - http://www.jcuda.org
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
 * 
 * 
 * --- BEGIN OF CUDPP LICENSE --->
 * Copyright (c) 2007-2010 The Regents of the University of California, Davis
 * campus ("The Regents") and NVIDIA Corporation ("NVIDIA"). All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice, 
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, 
 *       this list of conditions and the following disclaimer in the documentation 
 *       and/or other materials provided with the distribution.
 *     * Neither the name of the The Regents, nor NVIDIA, nor the names of its 
 *       contributors may be used to endorse or promote products derived from this 
 *       software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * <--- END OF CUDPP LICENSE ---
 * 
 * Homepage for CUDPP: http://code.google.com/p/cudpp
 */


package jcuda.jcudpp;
import jcuda.*;

/**
 * Java bindings for the public interface of CUDPP, the CUDA Data 
 * Parallel Primitives Library. <br>
 * (http://code.google.com/p/cudpp) <br>
 * <br>
 * Most comments are taken from the CUDPP library documentation.<br>
 */
public class JCudpp
{
    /**
     * The value that indicates that a hash key was not found
     */
    // This is implicitly defined via this line from cudpp_hash.cpp 
    // const unsigned int CUDPP_HASH_KEY_NOT_FOUND = CudaHT::CuckooHashing::kNotFound;
    // referring to this value from definitions.h:
    // const unsigned kNotFound = 0xffffffffu; //!< Signifies that a query key was not found.
    public static final int CUDPP_HASH_KEY_NOT_FOUND = 0xFFFFFFFF;
    
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not CUDPPResult.CUDPP_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCudpp()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Set the specified log level for the JCudpp library.<br>
     * <br>
     * Currently supported log levels:
     * <br>
     * LOG_QUIET: Never print anything <br>
     * LOG_ERROR: Print error messages <br>
     * LOG_TRACE: Print a trace of all native function calls <br>
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevel(logLevel.ordinal());
    }

    private static native void setLogLevel(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only return the CUDPPResult error code from the underlying CUDA function.
     * If exceptions are enabled, a CudaException with a detailed error 
     * message will be thrown if a method is about to return a result code 
     * that is not CUDPPResult.CUDPP_SUCCESS
     * 
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to CUDPPResult.CUDPP_SUCCESS and
     * exceptions have been enabled, this method will throw a 
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     * 
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not CUDPPResult.CUDPP_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != CUDPPResult.CUDPP_SUCCESS)
        {
            throw new CudaException(CUDPPResult.stringFor(result));
        }
        return result;
    }



    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly by the user of
     * the library: The library will automatically be 
     * initialized with the first function call.
     */
    public static void initialize()
    {
        assertInit();
    }


    /**
     * Assert that the native library that corresponds to the current value
     * of the emulation mode flag has been loaded.
     */
    private static void assertInit()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCudpp");
            initialized = true;
        }
    }

    /**
     * Creates an instance of the CUDPP library, and returns a handle.<br>
     * <br>
     * cudppCreate() must be called before any other CUDPP function. In a 
     * multi-GPU application that uses multiple CUDA context, cudppCreate() 
     * must be called once for each CUDA context. Each call returns a 
     * different handle, because each CUDA context (and the host thread 
     * that owns it) must use a separate instance of the CUDPP library.
     * <br>
     * Parameters:<br>
     * <pre>
     *  [in,out]    theCudpp    a pointer to the CUDPPHandle for the created CUDPP instance.
     * </pre>
     * @param theCudpp The handle
     * @return CUDPPResult indicating success or error condition
     */
    public static int cudppCreate(CUDPPHandle theCudpp)
    {
        return checkResult(cudppCreateNative(theCudpp));
    }
    private static native int cudppCreateNative(CUDPPHandle theCudpp);

    /** 
     * Destroys an instance of the CUDPP library given its handle.<br>
     * <br>
     * cudppDestroy() should be called once for each handle created 
     * using cudppCreate(), to ensure proper resource cleanup of all 
     * library instances.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    theCudpp    the handle to the CUDPP instance to destroy.
     * </pre>
     * @param theCudpp The handle
     * @return CUDPPResult indicating success or error condition
     */
    public static int cudppDestroy(CUDPPHandle theCudpp)
    {
        return checkResult(cudppDestroyNative(theCudpp));
    }
    private static native int cudppDestroyNative(CUDPPHandle theCudpp);


    /**
     * Create a CUDPP plan.<br>
     * <br>
     * A plan is a data structure containing state and intermediate storage
     * space that CUDPP uses to execute algorithms on data. A plan is created
     * by passing to cudppPlan() a CUDPPConfiguration that specifies the
     * algorithm, operator, datatype, and options. The size of the data must
     * also be passed to cudppPlan(), in the numElements, numRows, and
     * rowPitch arguments. These sizes are used to allocate internal storage
     * space at the time the plan is created. The CUDPP planner may use the
     * sizes, options, and information about the present hardware to choose
     * optimal settings.<br>
     * <br>
     * Note that numElements is the maximum size of the array to be processed
     * with this plan. That means that a plan may be re-used to process
     * (for example, to sort or scan) smaller arrays.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    cudppHandle A handle to an instance of the CUDPP library used for resource management 
     *     [out]   planHandle  A pointer to an opaque handle to the internal plan
     *     [in]    config  The configuration struct specifying algorithm and options
     *     [in]    numElements     The maximum number of elements to be processed
     *     [in]    numRows     The number of rows (for 2D operations) to be processed
     *     [in]    rowPitch    The pitch of the rows of input data, in elements
     * </pre>
     */
    public static int cudppPlan(
        CUDPPHandle cudppHandle,
        CUDPPHandle planHandle,
        CUDPPConfiguration config,
        long n,
        long rows,
        long rowPitch)
    {
        return checkResult(cudppPlanNative(cudppHandle, planHandle, config, n, rows, rowPitch));
    }
    private static native int cudppPlanNative(
        CUDPPHandle cudppHandle,
        CUDPPHandle planHandle,
        CUDPPConfiguration config,
        long n,
        long rows,
        long rowPitch);




    /**
     * Destroy a CUDPP Plan.<br>
     * <br>
     * Deletes the plan referred to by planHandle and all associated internal
     * storage.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *      [in]    planHandle  The CUDPPHandle to the plan to be destroyed
     * </pre>
     */
    public static int cudppDestroyPlan(CUDPPHandle plan)
    {
        return checkResult(cudppDestroyPlanNative(plan));
    }
    private static native int cudppDestroyPlanNative(CUDPPHandle plan);


    /**
     * Performs a scan operation of numElements on its input in GPU memory
     * (d_in) and places the output in GPU memory (d_out), with the scan
     * parameters specified in the plan pointed to by planHandle.<br>
     *
     * The input to a scan operation is an input array, a binary
     * associative operator (like + or max), and an identity element for
     * that operator (+'s identity is 0). The output of scan is the same
     * size as its input. Informally, the output at each element is the
     * result of operator applied to each input that comes before it.
     * For instance, the output of sum-scan at each element is the sum
     * of all the input elements before that input.<br>
     * <br>
     * More formally, for associative operator +,
     * out<sub>i</sub> = in<sub>0</sub> + in<sub>1</sub> + ... + in<sub>i-1</sub>.
     * <br>
     * <br>
     * CUDPP supports "exclusive" and "inclusive" scans. For the ADD operator,
     * an exclusive scan computes the sum of all input elements before the
     * current element, while an inclusive scan computes the sum of all input
     * elements up to and including the current element.<br>
     * <br>
     * Before calling scan, create an internal plan using cudppPlan().
     * <br>
     * After you are finished with the scan plan, clean up with
     * cudppDestroyPlan().<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    planHandle  Handle to plan for this scan
     *     [out]   d_out   output of scan, in GPU memory
     *     [in]    d_in    input to scan, in GPU memory
     *     [in]    numElements     number of elements to scan
     * </pre>
     * @see jcuda.jcudpp.JCudpp#cudppPlan
     * @see jcuda.jcudpp.JCudpp#cudppDestroyPlan
     */
    public static int cudppScan(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long numElements)
    {
        return checkResult(cudppScanNative(planHandle, d_out, d_in, numElements));
    }
    private static native int cudppScanNative(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long numElements);


    /**
     * Performs numRows parallel scan operations of numElements each on its
     * input (d_in) and places the output in d_out, with the scan parameters
     * set by config. Exactly like cudppScan except that it runs on multiple
     * rows in parallel.<br>
     * <br>
     * Note that to achieve good performance with cudppMultiScan one should
     * allocate the device arrays passed to it so that all rows are aligned
     * to the correct boundaries for the architecture the app is running on.
     * The easy way to do this is to use cudaMallocPitch() to allocate a 2D
     * array on the device. Use the rowPitch parameter to cudppPlan() to
     * specify this pitch. The easiest way is to pass the device pitch
     * returned by cudaMallocPitch to cudppPlan() via rowPitch.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    planHandle  handle to CUDPPScanPlan
     *     [out]   d_out   output of scan, in GPU memory
     *     [in]    d_in    input to scan, in GPU memory
     *     [in]    numElements     number of elements (per row) to scan
     *     [in]    numRows     number of rows to scan in parallel
     * </pre>
     * @see jcuda.jcudpp.JCudpp#cudppScan
     * @see jcuda.jcudpp.JCudpp#cudppPlan
     */
    public static int cudppMultiScan(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long numElements,
        long numRows)
    {
        return checkResult(cudppMultiScanNative(planHandle, d_out, d_in, numElements, numRows));
    }
    private static native int cudppMultiScanNative(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long numElements,
        long numRows);


    /**
     * Performs a segmented scan operation of numElements on its input in GPU
     * memory (d_idata) and places the output in GPU memory (d_out), with the
     * scan parameters specified in the plan pointed to by planHandle.<br>
     * <br>
     * The input to a segmented scan operation is an input array of data, an
     * input array of flags which demarcate segments, a binary associative
     * operator (like + or max), and an identity element for that operator
     * (+'s identity is 0). The array of flags is the same length as the input
     * with 1 marking the the first element of a segment and 0 otherwise.
     * The output of segmented scan is the same size as its input. Informally,
     * the output at each element is the result of operator applied to each
     * input that comes before it in that segment. For instance, the output
     * of segmented sum-scan at each element is the sum of all the input
     * elements before that input in that segment.<br>
     * <br>
     * More formally, for associative operator +,
     * out<sub>i</sub> = in<sub>k</sub> + in<sub>k+1</sub> + ... + in<sub>i-1</sub>. k is the index of the first element
     * of the segment in which i lies.<br>
     * <br>
     * We support both "exclusive" and "inclusive" variants. For a segmented
     * sum-scan, the exclusive variant computes the sum of all input elements
     * before the current element in that segment, while the inclusive variant
     * computes the sum of all input elements up to and including the current
     * element, in that segment.<br>
     * <br>
     * Before calling segmented scan, create an internal plan using cudppPlan().
     * <br>
     * After you are finished with the scan plan, clean up with cudppDestroyPlan().
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    planHandle  Handle to plan for this scan
     *     [out]   d_out   output of segmented scan, in GPU memory
     *     [in]    d_idata     input data to segmented scan, in GPU memory
     *     [in]    d_iflags    input flags to segmented scan, in GPU memory
     *     [in]    numElements     number of elements to perform segmented scan on
     * </pre>
     * @see jcuda.jcudpp.JCudpp#cudppPlan
     * @see jcuda.jcudpp.JCudpp#cudppDestroyPlan
     */

    public static int cudppSegmentedScan(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_idata,
        Pointer d_iflags,
        long numElements)
    {
        return checkResult(cudppSegmentedScanNative(planHandle, d_out, d_idata, d_iflags, numElements));
    }

    private static native int cudppSegmentedScanNative(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_idata,
        Pointer d_iflags,
        long numElements);



    /**
     * Given an array d_in and an array of 1/0 flags in deviceValid, returns
     * a compacted array in d_out of corresponding only the "valid" values
     * from d_in.<br>
     * <br>
     * Takes as input an array of elements in GPU memory (d_in) and an
     * equal-sized unsigned int array in GPU memory (deviceValid) that
     * indicate which of those input elements are valid. The output is a
     * packed array, in GPU memory, of only those elements marked as valid.
     * <br>
     * Internally, uses cudppScan.<br>
     * <br>
     * Example:<br>
     * <pre>
     *  d_in    = [ a b c d e f ]
     *  deviceValid = [ 1 0 1 1 0 1 ]
     *  d_out   = [ a c d f ]
     * </pre>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    planHandle  handle to CUDPPCompactPlan
     *     [out]   d_out   compacted output
     *     [out]   d_numValidElements  set during cudppCompact;
     *             is set with the number of elements valid flags in the
     *             d_isValid input array
     *     [in]    d_in    input to compact
     *     [in]    d_isValid   which elements in d_in are valid
     *     [in]    numElements     number of elements in d_in
     * </pre>
     */
    public static int cudppCompact(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_numValidElements,
        Pointer d_in,
        Pointer d_isValid,
        long numElements)
    {
        return checkResult(cudppCompactNative(planHandle, d_out, d_numValidElements, d_in, d_isValid, numElements));
    }

    private static native int cudppCompactNative(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_numValidElements,
        Pointer d_in,
        Pointer d_isValid,
        long numElements);


    /**
     * Reduces an array to a single element using a binary associative operator. <br>
     * <br>
     * For example, if the operator is CUDPP_ADD, then:
     * <pre>
     * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
     * d_out   = [ 6 ]
     * </pre>
     * If the operator is CUDPP_MIN, then:
     * <pre>
     * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
     * d_out   = [ -4 ]
     * </pre>
     * Limits: numElements must be at least 1, and is currently limited only by 
     * the addressable memory in CUDA (and the output accuracy is limited by 
     * numerical precision). <br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    planHandle  handle to CUDPPReducePlan
     * [out]   d_out   Output of reduce (a single element) in GPU memory. Must be a pointer to an array of at least a single element.
     * [in]    d_in    Input array to reduce in GPU memory. Must be a pointer to an array of at least numElements elements.
     * [in]    numElements the number of elements to reduce.
     * </pre>
     * @return CUDPPResult indicating success or error condition
     */
    public static int cudppReduce(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long  numElements)
    {
        return checkResult(cudppReduceNative(planHandle, d_out, d_in, numElements));
    }
    private static native int cudppReduceNative(
        CUDPPHandle planHandle,
        Pointer d_out,
        Pointer d_in,
        long numElements);

    /**
     * Sorts key-value pairs or keys only.<br>
     * <br>
     * Takes as input an array of keys in GPU memory (d_keys) and an optional 
     * array of corresponding values, and outputs sorted arrays of keys and 
     * (optionally) values in place. Key-value and key-only sort is selected 
     * through the configuration of the plan, using the options 
     * CUDPP_OPTION_KEYS_ONLY and CUDPP_OPTION_KEY_VALUE_PAIRS.<br>
     * <br>
     * Supported key types are CUDPP_FLOAT and CUDPP_UINT. Values can be any
     * 32-bit type (internally, values are treated only as a payload and cast 
     * to unsigned int).<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *    [in]    planHandle  handle to CUDPPSortPlan
     *    [out]   d_keys  keys by which key-value pairs will be sorted
     *    [in]    d_values    values to be sorted
     *    [in]    numElements     number of elements in d_keys and d_values
     * </pre>
     * @see jcuda.jcudpp.JCudpp#cudppPlan
     * @see jcuda.jcudpp.CUDPPConfiguration
     * @see jcuda.jcudpp.CUDPPAlgorithm
     */
    public static int cudppRadixSort(
        CUDPPHandle planHandle,
        Pointer d_keys,
        Pointer d_values,
        long numElements)
    {
        return checkResult(cudppRadixSortNative(planHandle, d_keys, d_values, numElements));
    }

    private static native int cudppRadixSortNative(
        CUDPPHandle planHandle,
        Pointer d_keys,
        Pointer d_values,
        long numElements);

    /**
     * Sorts key-value pairs or keys only.<br>
     * <br>
     * Takes as input an array of keys in GPU memory (d_keys) and an 
     * optional array of corresponding values, and outputs sorted 
     * arrays of keys and (optionally) values in place. Radix sort or 
     * Merge sort is selected through the configuration (.algorithm) 
     * Key-value and key-only sort is selected through the configuration 
     * of the plan, using the options CUDPP_OPTION_KEYS_ONLY and 
     * CUDPP_OPTION_KEY_VALUE_PAIRS.<br>
     * <br>
     * Supported key types are CUDPP_FLOAT and CUDPP_UINT. Values can 
     * be any 32-bit type (internally, values are treated only as a 
     * payload and cast to unsigned int).<br>
     * <br>
     * Parameters<br>
     * <pre>
     * [in]    planHandle  handle to CUDPPSortPlan
     * [out]   d_keys  keys by which key-value pairs will be sorted
     * [in]    d_values    values to be sorted
     * [in]    numElements number of elements in d_keys and d_values
     * </pre>
     */
    public static int cudppMergeSort(
        CUDPPHandle planHandle,
        Pointer d_keys, 
        Pointer d_values,                                                                       
        long numElements)
    {
        return checkResult(cudppMergeSortNative(planHandle, d_keys, d_values, numElements));
    }
    
    private static native int cudppMergeSortNative(
        CUDPPHandle planHandle,
        Pointer d_keys, 
        Pointer d_values,                                                                       
        long numElements);
    

    /**
     * Sorts strings. 
     * <br>
     * Keys are the first four characters of the string, and values 
     * are the addresses where the strings reside in memory (stringVals)<br>
     * <br>
     * Takes as input an array of strings arranged as a char* array 
     * with NULL terminating characters. This function will reformat 
     * this info into keys (first four chars) values(pointers to 
     * string array addresses) and aligned string value array.<br>
     * <br>
     * Parameters</br>
     * <pre>
     * [in]    planHandle  handle to CUDPPSortPlan
     * [in]    stringVals  Original string input, no need for alignment or offsets.
     * [in]    d_address   Pointers (in order) to each strings starting location in the stringVals array
     * [in]    termC   Termination character used to separate strings
     * [in]    numElements number of strings
     * [in]    stringArrayLength   Length in uint of the size of all strings
     * </pre>
     */
    public static int cudppStringSort(
        CUDPPHandle planHandle,                          
        Pointer d_stringVals,
        Pointer d_address,
        byte termC,
        long numElements,
        long stringArrayLength)
    {
        return checkResult(cudppStringSortNative(planHandle, d_stringVals, d_address, termC, numElements, stringArrayLength));
    }
    private static native int cudppStringSortNative(
        CUDPPHandle planHandle,                          
        Pointer d_stringVals,
        Pointer d_address,
        byte termC,
        long numElements,
        long stringArrayLength);

    /**
     * Create a CUDPP Sparse Matrix Object.<br>
     * <br>
     * The sparse matrix plan is a data structure containing state and
     * intermediate storage space that CUDPP uses to perform sparse
     * matrix dense vector multiply. This plan is created by passing to
     * CUDPPSparseMatrixVectorMultiplyPlan() a CUDPPConfiguration that
     * specifies the algorithm (sprarse matrix-dense vector multiply)
     * and datatype, along with the sparse matrix itself in CSR format.
     * The number of non-zero elements in the sparse matrix must also
     * be passed as numNonZeroElements. This is used to allocate internal
     * storage space at the time the sparse matrix plan is created.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    cudppHandle A handle to an instance of the CUDPP library used for resource management 
     *     [out]   sparseMatrixHandle  A pointer to an opaque handle to the sparse matrix object
     *     [in]    config  The configuration struct specifying algorithm and options
     *     [in]    numNonZeroElements  The number of non zero elements in the sparse matrix
     *     [in]    numRows     This is the number of rows in y, x and A for y = A * x
     *     [in]    A   The matrix data
     *     [in]    h_rowIndices    An array containing the index of the start of each row in A
     *     [in]    h_indices   An array containing the index of each nonzero element in A
     * </pre>
     */
    public static int cudppSparseMatrix(
        CUDPPHandle cudppHandle,
        CUDPPHandle sparseMatrixHandle,
        CUDPPConfiguration config,
        long numNonZeroElements,
        long numRows,
        Pointer A,
        Pointer h_rowIndices,
        Pointer h_indices)
    {
        return checkResult(cudppSparseMatrixNative(cudppHandle, sparseMatrixHandle, config, numNonZeroElements, numRows, A, h_rowIndices, h_indices));
    }
    private static native int cudppSparseMatrixNative(
        CUDPPHandle cudppHandle, 
        CUDPPHandle sparseMatrixHandle,
        CUDPPConfiguration config,
        long numNonZeroElements,
        long numRows,
        Pointer A,
        Pointer h_rowIndices,
        Pointer h_indices);



    /**
     * Destroy a CUDPP Sparse Matrix Object.<br>
     * <br>
     * Deletes the sparse matrix data and plan referred to by
     * sparseMatrixHandle and all associated internal storage.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *     [in]    sparseMatrixHandle  The CUDPPHandle to the matrix object to be destroyed
     * </pre>
     */
    public static int cudppDestroySparseMatrix(
        CUDPPHandle sparseMatrixHandle)
    {
        return checkResult(cudppDestroySparseMatrixNative(sparseMatrixHandle));
    }

    private static native int cudppDestroySparseMatrixNative(
        CUDPPHandle sparseMatrixHandle);



    /**
     * Perform matrix-vector multiply y = A*x for arbitrary sparse matrix
     * A and vector x.<br>
     * <br>
     * Given a matrix object handle (which has been initialized using
     * cudppSparseMatrix()), This function multiplies the input vector
     * d_x by the matrix referred to by sparseMatrixHandle, returning
     * the result in d_y.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *         sparseMatrixHandle  Handle to a sparse matrix object created
     *                             with cudppSparseMatrix()
     *         d_y     The output vector, y
     *         d_x     The input vector, x
     * </pre>
     * @see jcuda.jcudpp.JCudpp#cudppSparseMatrix
     * @see jcuda.jcudpp.JCudpp#cudppDestroySparseMatrix
     */
    public static int cudppSparseMatrixVectorMultiply(
        CUDPPHandle sparseMatrixHandle,
        Pointer d_y, Pointer d_x)
    {
        return checkResult(cudppSparseMatrixVectorMultiplyNative(sparseMatrixHandle, d_y, d_x));
    }

    private static native int cudppSparseMatrixVectorMultiplyNative(
        CUDPPHandle sparseMatrixHandle,
        Pointer d_y, Pointer d_x);


    /**
     * Rand puts numElements random 32-bit elements into d_out.<br>
     * <br>
     * Outputs numElements random values to d_out. d_out must be of 
     * type unsigned int, allocated in device memory.<br>
     * <br>
     * The algorithm used for the random number generation is 
     * stored in planHandle. Depending on the specification of the pseudo 
     * random number generator(PRNG), the generator may have one or more 
     * seeds. To set the seed, use cudppRandSeed().<br>
     * <br>
     * Currently only MD5 PRNG is supported. We may provide more 
     * rand routines in the future.
     * Parameters:<br>
     * <pre>
     *    [in]    planHandle  Handle to plan for rand
     *    [in]    numElements     number of elements in d_out.
     *    [out]   d_out   output of rand, in GPU memory. Should be an array of unsigned integers.
     * </pre>
     * 
     * @see jcuda.jcudpp.CUDPPConfiguration
     * @see jcuda.jcudpp.CUDPPAlgorithm
     */
    public static int cudppRand(CUDPPHandle planHandle, Pointer d_out, long numElements)
    {
        return checkResult(cudppRandNative(planHandle, d_out, numElements));
    }
    private static native int cudppRandNative(CUDPPHandle planHandle, Pointer d_out, long numElements);

    /**
     * Sets the seed used for rand.<br>
     * <br>
     * The seed is crucial to any random number generator as it allows a 
     * sequence of random numbers to be replicated. Since there may be 
     * multiple different rand algorithms in CUDPP, cudppRandSeed uses 
     * planHandle to determine which seed to set. Each rand algorithm 
     * has its own unique set of seeds depending on what the algorithm 
     * needs.<br>
     * <br>
     * Parameters:<br>
     * <pre>
     *    [in]    planHandle  the handle to the plan which specifies which rand seed to set
     *    [in]    seed    the value which the internal cudpp seed will be set to 
     */
    public static int cudppRandSeed(CUDPPHandle planHandle, int seed)
    {
        return checkResult(cudppRandSeedNative(planHandle, seed));
    }
    private static native int cudppRandSeedNative(CUDPPHandle planHandle, int seed);

    
    
    /**
     * Solves tridiagonal linear systems.<br>
     * <br>
     * The solver uses a hybrid CR-PCR algorithm described in our papers 
     * "Fast Fast Tridiagonal Solvers on the GPU" and "A Hybrid Method for 
     * Solving Tridiagonal Systems on the GPU". (See the References bibliography). 
     * Please refer to the papers for a complete description of the basic CR 
     * (Cyclic Reduction) and PCR (Parallel Cyclic Reduction) algorithms and 
     * their hybrid variants.<br>
     * <br>
     * Both float and double data types are supported.<br>
     * Both power-of-two and non-power-of-two system sizes are supported.<br>
     * The maximum system size could be limited by the maximum number of threads 
     * of a CUDA block, the number of registers per multiprocessor, and the amount 
     * of shared memory available. For example, on the GTX 280 GPU, the maximum 
     * system size is 512 for the float datatype, and 256 for the double datatype, 
     * which is limited by the size of shared memory in this case.<br>
     * The maximum number of systems is 65535, that is the maximum number of 
     * one-dimensional blocks that could be launched in a kernel call. Users 
     * could launch the kernel multiple times to solve more systems if required.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    planHandle  Handle to plan for tridiagonal solver
     * [in]    d_a Lower diagonal
     * [in]    d_b Main diagonal
     * [in]    d_c Upper diagonal
     * [in]    d_d Right hand side
     * [out]   d_x Solution vector
     * [in]    systemSize  The size of the linear system
     * [in]    numSystems  The number of systems to be solved
     * <pre>
     * @return CUDPPResult indicating success or error condition
     */
    public static int cudppTridiagonal(
        CUDPPHandle planHandle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer d, 
        Pointer x, 
        int systemSize, 
        int numSystems)
    {
        return checkResult(cudppTridiagonalNative(planHandle, a, b, c, d, x, systemSize, numSystems));
    }
    private static native int cudppTridiagonalNative(
        CUDPPHandle planHandle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer d, 
        Pointer x, 
        int systemSize, 
        int numSystems);
 
    /**
     * Compresses data stream.<br>
     * <br>
     * Performs compression using a three stage pipeline consisting of 
     * the Burrows-Wheeler transform, the move-to-front transform, and 
     * Huffman encoding. The compression algorithms are described in 
     * our paper "Parallel Lossless Data Compression on the GPU". 
     * (See the References bibliography).<br>
     * <br>
     * <ul>
     *   <li>Only unsigned char type is supported.</li>
     *   <li>Currently, the input stream (d_uncompressed) must be a 
     *   buffer of 1,048,576 (uchar) elements (~1MB).</li>
     *   <li>The BWT Index (d_bwtIndex) is an integer number (int). 
     *   This is used during the reverse-BWT stage.</li>
     *   <li>The Histogram size pointer (d_histSize) can be ignored
     *    and can be passed a null pointer.</li>
     *   <li>The Histrogram (d_hist) is a 256-entry (unsigned int) 
     *   buffer. The histogram is used to construct the Huffman tree 
     *   during decoding.</li>
     *   <li>The Encoded offset table (d_encodeOffset) is a 256-entry 
     *   (unsigned int) buffer. Since the input stream is compressed 
     *   in blocks of 4096 characters, the offset table gives the starting 
     *   offset of where each block starts in the compressed data 
     *   (d_compressedSize). The very first uint at each starting offset 
     *   gives the size (in words) of that corresponding compressed block. 
     *   This allows us to decompress each 4096 character-block in 
     *   parallel.</li>
     *   <li>The size of compressed data (d_compressedSize) is a uint and 
     *   gives the final size (in words) of the compressed data.</li>
     *   <li>The compress data stream (d_compressed) is a uint buffer. 
     *   The user should allocate enough memory for worst-case 
     *   (no compression occurs).</li>
     *   <li>numElements is a uint and must be set to 1048576.</li>
     * </ul>
     * <br>
     * Parameters<br>
     * <pre>
     * [out]   d_bwtIndex  BWT Index (int)
     * [out]   d_histSize  Histogram size (ignored, null ptr)
     * [out]   d_hist  Histogram (256-entry, uint)
     * [out]   d_encodeOffset  Encoded offset table (256-entry, uint)
     * [out]   d_compressedSize    Size of compressed data (uint)
     * [out]   d_compressed    Compressed data
     * [in]    planHandle  Handle to plan for compressor
     * [in]    d_uncompressed  Uncompressed data
     * [in]    numElements Number of elements to compress
     * </pre> 
     */
    public static int cudppCompress(
        CUDPPHandle planHandle,
        Pointer d_uncompressed,
        Pointer d_bwtIndex,
        Pointer d_histSize,
        Pointer d_hist,
        Pointer d_encodeOffset,
        Pointer d_compressedSize,
        Pointer d_compressed,
        long numElements)
    {
        return checkResult(cudppCompressNative(planHandle, d_uncompressed, d_bwtIndex, d_histSize, d_hist, d_encodeOffset, d_compressedSize, d_compressed, numElements));
    }
    private static native int cudppCompressNative(
        CUDPPHandle planHandle,
        Pointer d_uncompressed,
        Pointer d_bwtIndex,
        Pointer d_histSize,
        Pointer d_hist,
        Pointer d_encodeOffset,
        Pointer d_compressedSize,
        Pointer d_compressed,
        long numElements);


    /**
     * Performs the Burrows-Wheeler Transform.<br>
     * <br>
     * Performs a parallel Burrows-Wheeler transform on 1,048,576 elements. 
     * The BWT leverages a string-sort algorithm based on merge-sort.<br>
     * <br>
     * <ul>
     *   <li>Currently, the BWT can only be performed on 1,048,576 (uchar) 
     *   elements.</li>
     *   <li>The transformed string is written to d_x.</li>
     *   <li>The BWT index (used during the reverse-BWT) is recorded as an 
     *   int in d_index.</li>
     * </ul>
     * Parameters<br>
     * <pre>
     * [in]    planHandle  Handle to plan for BWT
     * [out]   d_in    BWT Index
     * [out]   d_out   Output data
     * [in]    d_index Input data
     * [in]    numElements Number of elements 
     * </pre>
     */
    public static int cudppBurrowsWheelerTransform(
        CUDPPHandle planHandle,
        Pointer d_in,
        Pointer d_out,
        Pointer d_index,
        long numElements)
    {
        return checkResult(cudppBurrowsWheelerTransformNative(planHandle, d_in, d_out, d_index, numElements));
    }
    private static native int cudppBurrowsWheelerTransformNative(
        CUDPPHandle planHandle,
        Pointer d_in,
        Pointer d_out,
        Pointer d_index,
        long numElements);

    
    /**
     * Performs the Move-to-Front Transform.<br>
     * <br>
     * Performs a parallel move-to-front transform on 1,048,576 elements. 
     * The MTF uses a scan-based algorithm to parallelize the computation. 
     * The MTF uses a scan-based algorithm described in our paper "Parallel 
     * Lossless Data Compression on the GPU". (See the References 
     * bibliography).<br>
     * <br>
     * <ul>
     *   <li>Currently, the MTF can only be performed on 1,048,576 
     *   (uchar) elements.</li>
     *   <li>The transformed string is written to d_mtfOut.</li>
     * </ul>
     * Parameters<br>
     * <pre>
     * [in]    planHandle  Handle to plan for MTF
     * [out]   d_out   Output data
     * [in]    d_in    Input data
     * [in]    numElements Number of elements
     * </pre> 
     */
    public static int cudppMoveToFrontTransform(
        CUDPPHandle planHandle,
        Pointer d_in,
        Pointer d_out,
        long numElements)
    {
        return checkResult(cudppMoveToFrontTransformNative(planHandle, d_in, d_out, numElements));
    }
    private static native int cudppMoveToFrontTransformNative(
        CUDPPHandle planHandle,
        Pointer d_in,
        Pointer d_out,
        long numElements);


    /**
     * Performs list ranking of linked list node values.<br>
     * <br>
     * Performs parallel list ranking on values of a linked-list using 
     * a pointer-jumping algorithm.<br>
     * <br>
     * Takes as input an array of values in GPU memory (d_unranked_values) 
     * and an equal-sized int array in GPU memory (d_next_indices) that 
     * represents the next indices of the linked list. The index of the 
     * head node (head) is given as an unsigned int. The output 
     * (d_ranked_values) is an equal-sized array, in GPU memory, that 
     * has the values ranked in-order.<br>
     * <br>
     * Example:
     * <pre><code>
     * d_a = [ f a c d b e ]
     * d_b = [ -1 4 3 5 2 0 ]
     * head = 1
     * d_x = [ a b c d e f ]
     * </code></pre>
     *<br>
     *Parameters<br>
     *<pre>
     * [in]    planHandle  Handle to plan for list ranking
     * [out]   d_ranked_values Output ranked values
     * [in]    d_unranked_values   Input unranked values
     * [in]    d_next_indices  Input next indices
     * [in]    head    Input head node index
     * [in]    numElements number of nodes
     * </pre> 
     */
    public static int cudppListRank(
        CUDPPHandle planHandle, 
        Pointer d_ranked_values,  
        Pointer d_unranked_values,
        Pointer d_next_indices,
        long head,
        long numElements)
    {
        return checkResult(cudppListRankNative(planHandle, d_ranked_values, d_unranked_values, d_next_indices, head, numElements));
    }
    private static native int cudppListRankNative(
        CUDPPHandle planHandle, 
        Pointer d_ranked_values,  
        Pointer d_unranked_values,
        Pointer d_next_indices,
        long head,
        long numElements);
    
    /**
     * Performs the Suffix Array.<br>
     * <br>
     * Performs a parallel suffix array using linear-time recursive 
     * skew algorithm. The SA leverages a suffix-sort algorithm based 
     * on divide and conquer.<br>
     * <br>
     * <ul> 
     *   <li>
     *     The SA is GPU memory bounded, it needs about seven times size of 
     *     input data. 
     *   </li>
     *   <li>
     *     Only unsigned char type is supported. 
     *   </li>
     *   <li>
     *     The input char array is transformed into an unsigned int array 
     *     storing the key values followed by three 0s for the convenience 
     *     of building triplets.
     *   </li>
     *   <li>
     *     The output data is an unsigned int array storing the positions 
     *     of the lexicographically sorted suffixes not including the last 
     *     {0,0,0} triplet.
     *   </li>
     * <br>
     * Parameters
     * <pre>
     * [in]    planHandle  Handle to plan for BWT
     * [out]   d_in    Input data
     * [out]   d_out   Output data
     * [in]    numElements Number of elements 
     * </pre>
     */
    public static int cudppSuffixArray(
        CUDPPHandle planHandle,
        Pointer d_str,
        Pointer d_keys_sa,
        long numElements)
    {
        return checkResult(cudppSuffixArrayNative(planHandle, d_str, d_keys_sa, numElements));
    }
    private static native int cudppSuffixArrayNative(
        CUDPPHandle planHandle,
        Pointer d_str,
        Pointer d_keys_sa,
        long numElements);

    
    /** 
     * Creates a CUDPP hash table in GPU memory given an input hash table 
     * configuration; returns the plan for that hash table. <br>
     * <br>
     * Requires a CUDPPHandle for the CUDPP instance (to ensure thread 
     * safety); call cudppCreate() to get this handle.<br>
     * <br>
     * The hash table implementation requires hardware capability 2.0 
     * or higher (64-bit atomic operations).<br>
     * <br>
     * Hash table types and input parameters are discussed in 
     * CUDPPHashTableType and CUDPPHashTableConfig.<br>
     * <br>
     * After you are finished with the hash table, clean up with 
     * cudppDestroyHashTable().<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    cudppHandle Handle to CUDPP instance
     * [out]   plan    Handle to hash table instance
     * [in]    config  Configuration for hash table to be created
     * </pre>
     * @see JCudpp#cudppCreate
     * @see JCudpp#cudppDestroyHashTable
     * @see CUDPPHashTableType
     * @see CUDPPHashTableConfig 
     * @return CUDPPResult indicating if creation was successful
     */
    public static int cudppHashTable(
        CUDPPHandle cudppHandle, 
        CUDPPHandle plan,
        CUDPPHashTableConfig config)
    {
        return checkResult(cudppHashTableNative(cudppHandle, plan, config));
    }
    private static native int cudppHashTableNative(
        CUDPPHandle cudppHandle, 
        CUDPPHandle plan,
        CUDPPHashTableConfig config);


    /** 
     * Destroys a hash table given its handle. <br>
     * <br>
     * Requires a CUDPPHandle for the CUDPP instance (to ensure thread safety); 
     * call cudppCreate() to get this handle.<br>
     * <br>
     * Requires a CUDPPHandle for the hash table instance; call cudppHashTable() 
     * to get this handle.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    cudppHandle Handle to CUDPP instance
     * [in]    plan    Handle to hash table instance
     * </pre>
     * @see JCudpp#cudppHashTable 
     * @return CUDPPResult indicating if destruction was successful
     */
    public static int cudppDestroyHashTable(
        CUDPPHandle cudppHandle,
        CUDPPHandle plan)
    {
        return checkResult(cudppDestroyHashTableNative(cudppHandle, plan));
    }
    private static native int cudppDestroyHashTableNative(
        CUDPPHandle cudppHandle,
        CUDPPHandle plan);

    
    
    /**
     * Inserts keys and values into a CUDPP hash table. <br>
     * <br>
     * Requires a CUDPPHandle for the hash table instance; call cudppHashTable() 
     * to create the hash table and get this handle.<br>
     * <br>
     * d_keys and d_values should be in GPU memory. These should be pointers to 
     * arrays of unsigned ints.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    plan    Handle to hash table instance
     * [in]    d_keys  GPU pointer to keys to be inserted
     * [in]    d_vals  GPU pointer to values to be inserted
     * [in]    num Number of keys/values to be inserted
     * </pre>
     * @see JCudpp#cudppHashTable
     * @see JCudpp#cudppHashRetrieve 
     * @return CUDPPResult indicating if insertion was successful
     */
    public static int cudppHashInsert(
        CUDPPHandle plan,
        Pointer d_keys, 
        Pointer d_vals,
        long num)
    {
        return checkResult(cudppHashInsertNative(plan, d_keys, d_vals, num));
    }
    private static native int cudppHashInsertNative(
        CUDPPHandle plan,
        Pointer d_keys, 
        Pointer d_vals,
        long num);

    
    
    /**
     * Retrieves values, given keys, from a CUDPP hash table. <br>
     * <br>
     * Requires a CUDPPHandle for the hash table instance; call cudppHashTable() 
     * to create the hash table and get this handle.<br>
     * <br>
     * d_keys and d_values should be in GPU memory. These should be pointers to 
     * arrays of unsigned ints.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    plan    Handle to hash table instance
     * [in]    d_keys  GPU pointer to keys to be retrieved
     * [out]   d_vals  GPU pointer to values to be retrieved
     * [in]    num Number of keys/values to be retrieved
     * </pre>
     * @see JCudpp#cudppHashTable
     * @return CUDPPResult indicating if retrieval was successful
     */
    public static int cudppHashRetrieve(
        CUDPPHandle plan, 
        Pointer d_keys, 
        Pointer d_vals, 
        long num)
    {
        return checkResult(cudppHashRetrieveNative(plan, d_keys, d_vals, num));
    }
    private static native int cudppHashRetrieveNative(
        CUDPPHandle plan, 
        Pointer d_keys, 
        Pointer d_vals, 
        long num);

    
    /**
     * Retrieves the size of the values array in a multivalue hash table. <br>
     * <br>
     * Only relevant for multivalue hash tables.<br>
     * <br>
     * Requires a CUDPPHandle for the hash table instance; call cudppHashTable() 
     * to get this handle.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    plan    Handle to hash table instance
     * [out]   size    Pointer to size of multivalue hash table
     * </pre>
     * @see JCudpp#cudppHashTable
     * @see JCudpp#cudppMultivalueHashGetAllValues 
     * @return CUDPPResult indicating if operation was successful
     */
    public static int cudppMultivalueHashGetValuesSize(
        CUDPPHandle plan, 
        int size[])
    {
        return checkResult(cudppMultivalueHashGetValuesSizeNative(plan, size));
    }
    private static native int cudppMultivalueHashGetValuesSizeNative(
        CUDPPHandle plan, 
        int size[]);


    
    /**
     * Retrieves a pointer to the values array in a multivalue hash table. <br>
     * <br>
     * Only relevant for multivalue hash tables.<br>
     * <br>
     * Requires a CUDPPHandle for the hash table instance; call cudppHashTable() 
     * to get this handle.<br>
     * <br>
     * Parameters:
     * <pre>
     * [in]    plan    Handle to hash table instance
     * [out]   d_vals  Pointer to pointer of values (in GPU memory)
     * </pre>
     * @see JCudpp#cudppHashTable
     * @see JCudpp#cudppMultivalueHashGetValuesSize 
     * @return CUDPPResult indicating if operation was successful
     */
    public static int cudppMultivalueHashGetAllValues(
        CUDPPHandle plan,
        Pointer d_vals)
    {
        return checkResult(cudppMultivalueHashGetAllValuesNative(plan, d_vals));
    }
    private static native int cudppMultivalueHashGetAllValuesNative(
        CUDPPHandle plan,
        Pointer d_vals);
    
}
