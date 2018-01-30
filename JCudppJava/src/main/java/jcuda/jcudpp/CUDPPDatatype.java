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
 * Datatypes supported by CUDPP algorithms.
 *
 * @see jcuda.jcudpp.CUDPPConfiguration
 * @see jcuda.jcudpp.JCudpp#cudppPlan
 */
public class CUDPPDatatype
{
    /**
     * Character type (C char) - Closest Java type: byte
     */
    public static final int CUDPP_CHAR = 0;  
    
    /**
     * Unsigned character (byte) type (C unsigned char) - Closest Java type: byte
     */
    public static final int CUDPP_UCHAR = 1;
    
    /**
     * Short integer type (C short)
     */
    public static final int CUDPP_SHORT = 2;
    
    /**
     * Short unsigned integer type (C unsigned short)
     */
    public static final int CUDPP_USHORT = 3;
    
    /**
     * Integer type (C int) - Closest Java type: int
     */
    public static final int CUDPP_INT = 4; 
    
    /**
     * Unsigned integer type (C unsigned int) - Closest Java type: int
     */
    public static final int CUDPP_UINT = 5;
    
    /**
     * Float type (C float) - Closest Java type: float 
     */
    public static final int CUDPP_FLOAT = 6;  
    
    /**
     * Double type (C double)
     */
    public static final int CUDPP_DOUBLE = 7;  
    
    /**
     * 64-bit integer type (C long long)
     */
    public static final int CUDPP_LONGLONG = 8;
    
    /**
     * 64-bit unsigned integer type (C unsigned long long)
     */
    public static final int CUDPP_ULONGLONG = 9;
    
    /**
     * Invalid datatype (must be last in list)
     */
    public static final int CUDPP_DATATYPE_INVALID = 10;
    
    /**
     * Returns the String identifying the given CUDPPDatatype
     * 
     * @param n The CUDPPDatatype
     * @return The String identifying the given CUDPPDatatype
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDPP_CHAR: return "CUDPP_CHAR";
            case CUDPP_UCHAR: return "CUDPP_UCHAR";
            case CUDPP_SHORT: return "CUDPP_SHORT";
            case CUDPP_USHORT: return "CUDPP_USHORT";
            case CUDPP_INT: return "CUDPP_INT";
            case CUDPP_UINT: return "CUDPP_UINT";
            case CUDPP_FLOAT: return "CUDPP_FLOAT";
            case CUDPP_DOUBLE: return "CUDPP_DOUBLE";
            case CUDPP_LONGLONG: return "CUDPP_LONGLONG";
            case CUDPP_ULONGLONG: return "CUDPP_ULONGLONG";
            case CUDPP_DATATYPE_INVALID: return "CUDPP_DATATYPE_INVALID";
        }
        return "INVALID CUDPPDatatype: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUDPPDatatype()
    {
    }
    
};
