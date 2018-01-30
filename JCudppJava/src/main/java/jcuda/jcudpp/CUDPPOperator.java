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
 * Operators supported by CUDPP algorithms (currently scan and segmented
 * scan).<br/>
 * <br/>
 * These are all binary associative operators.
 *
 * @see jcuda.jcudpp.CUDPPConfiguration
 * @see jcuda.jcudpp.JCudpp#cudppPlan
 */
public class CUDPPOperator
{
    /**
     * Addition of two operands
     */
    public static final int CUDPP_ADD = 0;  
    
    /**
     * Multiplication of two operands
     */
    public static final int CUDPP_MULTIPLY = 1;
    
    /**
     * Minimum of two operands
     */
    public static final int CUDPP_MIN = 2; 
    
    /**
     * Maximum of two operands
     */
    public static final int CUDPP_MAX = 3;  
    
    /**
     * Invalid operator (must be last in list)
     */
    public static final int CUDPP_OPERATOR_INVALID = 4;
    
    /**
     * Returns the String identifying the given CUDPPOperator
     * 
     * @param n The CUDPPOperator
     * @return The String identifying the given CUDPPOperator
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDPP_ADD: return "CUDPP_ADD";  
            case CUDPP_MULTIPLY: return "CUDPP_MULTIPLY";  
            case CUDPP_MIN: return "CUDPP_MIN";  
            case CUDPP_MAX: return "CUDPP_MAX";  
            case CUDPP_OPERATOR_INVALID: return "CUDPP_OPERATOR_INVALID";  
        }
        return "INVALID CUDPPOperator: "+n;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPOperator()
    {
    }
    
};
