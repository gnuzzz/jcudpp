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
 * Configuration struct for creating a hash table.
 * 
 * @see jcuda.jcudpp.JCudpp#cudppHashTable 
 * @see CUDPPHashTableType
 */
public class CUDPPHashTableConfig
{
    /** 
     * The hash table type.
     * @see CUDPPHashTableType 
     */
    public int type;    

    /**
     * Number of elements to be stored in hash table 
     */
    public int kInputSize;    
    
    /**
     * Space factor multiple for the hash table; multiply space_usage by
     * kInputSize to get the actual space allocation in GPU memory. 
     * 1.05 is about the minimum possible to get a working hash table. 
     * Larger values use more space but take less time to construct. 
     */
    public float space_usage;          

    /**
     * Creates a new, uninitialized CUDPPHashTableConfig
     */
    public CUDPPHashTableConfig()
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
        return "CUDPPHashTableConfig["+
            CUDPPHashTableType.stringFor(type)+","+
            "kInputSize="+kInputSize+","+
            "space_usage="+space_usage+"]";
    }
}
