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
 * Options for configuring CUDPP algorithms.
 * 
 * @see jcuda.jcudpp.CUDPPConfiguration
 * @see jcuda.jcudpp.JCudpp#cudppPlan
 */
public class CUDPPOption
{
    /** 
     * Algorithms operate forward: from start to end of input array 
     */
    public static final int CUDPP_OPTION_FORWARD = 0x1;

    /** 
     * Algorithms operate backward: from end to start of array 
     */
    public static final int CUDPP_OPTION_BACKWARD = 0x2;

    /**
     * Exclusive (for scans) - scan includes all elements up to
     * (but not including) the current element
     */
    public static final int CUDPP_OPTION_EXCLUSIVE = 0x4;

    /**
     * Inclusive (for scans) - scan includes all elements up to
     * and including the current element
     */
    public static final int CUDPP_OPTION_INCLUSIVE = 0x8;

    /**
     * Algorithm performed only on the CTAs (blocks) with no communication
     * between blocks.<br />
     * <br />
     * TODO: Currently ignored
     */
    public static final int CUDPP_OPTION_CTA_LOCAL = 0x10;

    /**
     * No associated value to a key  (for global radix sort)
     */
    public static final int CUDPP_OPTION_KEYS_ONLY = 0x20;

    /**
     * Each key has an associated value
     */
    public static final int CUDPP_OPTION_KEY_VALUE_PAIRS = 0x40;
    
    /**
     * Returns the String identifying the given CUDPPOptions
     * 
     * @param n The CUDPPOptions
     * @return The String identifying the given CUDPPOptions
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "(no CUDPPOption)";
        }
        String result = "";
        if ((n & CUDPP_OPTION_FORWARD) != 0) result += "CUDPP_OPTION_FORWARD ";
        if ((n & CUDPP_OPTION_BACKWARD) != 0) result += "CUDPP_OPTION_BACKWARD ";
        if ((n & CUDPP_OPTION_EXCLUSIVE) != 0) result += "CUDPP_OPTION_EXCLUSIVE ";
        if ((n & CUDPP_OPTION_INCLUSIVE) != 0) result += "CUDPP_OPTION_INCLUSIVE";
        if ((n & CUDPP_OPTION_CTA_LOCAL) != 0) result += "CUDPP_OPTION_CTA_LOCAL ";
        if ((n & CUDPP_OPTION_KEYS_ONLY) != 0) result += "CUDPP_OPTION_KEYS_ONLY ";
        if ((n & CUDPP_OPTION_KEY_VALUE_PAIRS) != 0) result += "CUDPP_OPTION_KEY_VALUE_PAIRS ";
        if (result.length() == 0)
        {
            return "INVALID CUDPPOption: "+n;
        }
        return result;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPOption()
    {
    }
    
}
