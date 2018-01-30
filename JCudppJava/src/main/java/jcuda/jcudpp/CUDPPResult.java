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
 */

package jcuda.jcudpp;

/**
 * CUDPP Result codes returned by CUDPP API functions.
 */
public class CUDPPResult
{
    /**
     * No error.
     */
    public static final int CUDPP_SUCCESS = 0;

    /**
     * Specified handle (for example, to a plan) is invalid.
     */
    public static final int CUDPP_ERROR_INVALID_HANDLE = 1;

    /**
     * Specified configuration is illegal.
     * For example, an invalid or illogical
     * combination of options.
     */
    public static final int CUDPP_ERROR_ILLEGAL_CONFIGURATION = 2;

    /** 
     * The plan is not configured properly. For example, passing a 
     * plan for scan to cudppSegmentedScan. 
     */
    public static final int CUDPP_ERROR_INVALID_PLAN = 3;
    
    /** 
     * The function could not complete due to insufficient resources 
     * (typically CUDA device resources such as shared memory)
     * for the specified problem size. 
     */    
    public static final int CUDPP_ERROR_INSUFFICIENT_RESOURCES = 4;

    /**
     * Unknown or untraceable error.
     */
    public static final int CUDPP_ERROR_UNKNOWN = 9999;

    /**
     * Internal JCudpp error
     */
    public static final int JCUDPP_INTERNAL_ERROR = 0x80000001;

    /**
     * Returns the String representation of the given result
     *
     * @param result The result
     * @return The String representation of the given result
     */
    public static String stringFor(int result)
    {
        switch (result)
        {
            case CUDPP_SUCCESS                      : return "CUDPP_SUCCESS";
            case CUDPP_ERROR_INVALID_HANDLE         : return "CUDPP_ERROR_INVALID_HANDLE";
            case CUDPP_ERROR_ILLEGAL_CONFIGURATION  : return "CUDPP_ERROR_ILLEGAL_CONFIGURATION";
            case CUDPP_ERROR_INVALID_PLAN           : return "CUDPP_ERROR_INVALID_PLAN";
            case CUDPP_ERROR_INSUFFICIENT_RESOURCES : return "CUDPP_ERROR_ILLEGAL_CONFIGURATION";
            case CUDPP_ERROR_UNKNOWN                : return "CUDPP_ERROR_UNKNOWN";
            case JCUDPP_INTERNAL_ERROR              : return "JCUDPP_INTERNAL_ERROR";
        }
        return "INVALID CUDPPResult: "+result;
    }
    
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPResult()
    {
    }
    
};
