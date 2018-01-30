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
 * Configuration struct used to specify algorithm, datatype, operator,
 * and options when creating a plan for CUDPP algorithms.
 *
 * @see jcuda.jcudpp.JCudpp#cudppPlan
 */
public class CUDPPConfiguration
{
    /**
     * The algorithm to be used
     * 
     * @see CUDPPAlgorithm
     */
    public int algorithm; 
    
    /**
     * The numerical operator to be applied
     * 
     * @see CUDPPOperator
     */
    public int op;
    
    /**
     * The datatype of the input arrays
     * 
     * @see CUDPPDatatype
     */
    public int datatype; 
    
    /**
     * Options to configure the algorithm
     * 
     * @see CUDPPOption
     */
    public int options;
    
    /**
     * Creates a new, uninitialized CUDPPConfiguration
     */
    public CUDPPConfiguration()
    {
    }
    
    /**
     * Returns a String representation of this object.
     * 
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUDPPConfiguration["+
            CUDPPAlgorithm.stringFor(algorithm)+","+
            CUDPPOperator.stringFor(op)+","+
            CUDPPDatatype.stringFor(datatype)+","+
            CUDPPOption.stringFor(options)+"]";
    }
};
