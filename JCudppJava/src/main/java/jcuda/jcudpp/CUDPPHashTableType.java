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
 * Supported types of hash tables. 
 *
 * @see jcuda.jcudpp.CUDPPHashTableConfig
 */
public class CUDPPHashTableType
{
    /**
     * Stores a single value per key. Input is expected to be 
     * a set of key-value pairs, where the keys are all unique.
     */
    public static final int CUDPP_BASIC_HASH_TABLE  = 0;

    /**
     * Assigns each key a unique identifier and allows O(1) translation 
     * between the key and the unique IDs. Input is a set of keys that 
     * may, or may not, be repeated.
     */
    public static final int CUDPP_COMPACTING_HASH_TABLE     = 1;

    /**
     * Can store multiple values for each key. Multiple values for 
     * the same key are represented by different key-value pairs in 
     * the input.
     */
    public static final int CUDPP_MULTIVALUE_HASH_TABLE     = 2;

    /**
     * Invalid hash table; flags error if used. 
     */
    public static final int CUDPP_INVALID_HASH_TABLE    = 3;
    
    /**
     * Returns the String identifying the given CUDPPHashTableType
     * 
     * @param n The CUDPPHashTableType
     * @return The String identifying the given CUDPPHashTableType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDPP_BASIC_HASH_TABLE: return "CUDPP_BASIC_HASH_TABLE";
            case CUDPP_COMPACTING_HASH_TABLE: return "CUDPP_COMPACTING_HASH_TABLE";
            case CUDPP_MULTIVALUE_HASH_TABLE: return "CUDPP_MULTIVALUE_HASH_TABLE";
            case CUDPP_INVALID_HASH_TABLE: return "CUDPP_INVALID_HASH_TABLE";
        }
        return "INVALID CUDPPHashTableType: "+n;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPHashTableType()
    {
    }
    
};
