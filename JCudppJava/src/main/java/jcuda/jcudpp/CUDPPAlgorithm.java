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
 * Algorithms supported by CUDPP.  Used to create appropriate plans
 * using cudppPlan.
 *
 * @see jcuda.jcudpp.CUDPPConfiguration
 * @see jcuda.jcudpp.JCudpp#cudppPlan
 */
public class CUDPPAlgorithm
{
    /**
     * Scan
     */
    public static final int CUDPP_SCAN = 0;

    /**
     * Segmented scan
     */
    public static final int CUDPP_SEGMENTED_SCAN = 1;

    /**
     * Compact
     */
    public static final int CUDPP_COMPACT = 2;

    /**
     * Reduction 
     */
    public static final int CUDPP_REDUCE = 3;

    /** 
     * Radix sort within chunks, merge sort to merge chunks together 
     */
    public static final int CUDPP_SORT_RADIX = 4;

    /**
     * Merge Sort
     */
    public static final int CUDPP_SORT_MERGE = 5; 
    
    /**
     * String Sort
     */
    public static final int CUDPP_SORT_STRING = 6;
    
    /**
     * Sparse matrix - vector multiplication
     */
    public static final int CUDPP_SPMVMULT = 7;

    /**
     * Pseudo Random Number Generator using MD5 hash algorithm
     */
    public static final int CUDPP_RAND_MD5 = 8;

    /**
     *  Tridiagonal solver algorithm
     */
    public static final int CUDPP_TRIDIAGONAL = 9;
    
    /**
     * Lossless data compression
     */
    public static final int CUDPP_COMPRESS = 10;
    
    /**
     * List ranking
     */
    public static final int CUDPP_LISTRANK = 11;
    
    /**
     * Burrows-Wheeler transform
     */
    public static final int CUDPP_BWT = 12;
    
    /**
     * Move-to-Front transform
     */
    public static final int CUDPP_MTF = 13;
    
    /**
     * Suffix Array algorithm
     */
    public static final int CUDPP_SA = 14;
    
    /** 
     * Placeholder at end of enum 
     */
    public static final int CUDPP_SORT_INVALID = 15;
    
    /**
     * Returns the String identifying the given CUDPPAlgorithm
     * 
     * @param n The CUDPPAlgorithm
     * @return The String identifying the given CUDPPAlgorithm
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDPP_SCAN: return "CUDPP_SCAN";
            case CUDPP_SEGMENTED_SCAN: return "CUDPP_SEGMENTED_SCAN";
            case CUDPP_COMPACT: return "CUDPP_COMPACT";
            case CUDPP_REDUCE: return "CUDPP_REDUCE";
            case CUDPP_SORT_RADIX: return "CUDPP_SORT_RADIX";
            case CUDPP_SORT_MERGE: return "CUDPP_SORT_MERGE";
            case CUDPP_SORT_STRING: return "CUDPP_SORT_STRING";
            case CUDPP_SPMVMULT: return "CUDPP_SPMVMULT";
            case CUDPP_RAND_MD5: return "CUDPP_RAND_MD5";
            case CUDPP_TRIDIAGONAL: return "CUDPP_TRIDIAGONAL";
            case CUDPP_COMPRESS: return "CUDPP_COMPRESS";
            case CUDPP_LISTRANK: return "CUDPP_LISTRANK";
            case CUDPP_BWT: return "CUDPP_BWT";
            case CUDPP_MTF: return "CUDPP_MTF";            
            case CUDPP_SA: return "CUDPP_SA";            
            case CUDPP_SORT_INVALID: return "CUDPP_SORT_INVALID";
        }
        return "INVALID CUDPPAlgorithm: "+n;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPAlgorithm()
    {
    }
    
};
