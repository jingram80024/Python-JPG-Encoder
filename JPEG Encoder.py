# This rudimentary JPEG encoder takes high level structure from
# "Reducible" youtube channel's video on JPEG compression concepts.
# Quantization table is JPEG standard for Quality factor of 50.
# This program encodes and then decodes to show image quality comparison.
# REFERENCES: https://www.youtube.com/watch?v=0me3guauqOU

# As of 2/21/22 this is to be used for luma (Y) channel black and white only.

# | LIBRARIES |
#------------------------------------------------------------------------------
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import math
from PIL import Image

# | CONSTANTS |
#------------------------------------------------------------------------------
pi = math.pi
# N is size of NxN chunk samples of input image used for DCT
N = 8
file_name = 'dirt.jpg'
#file_name = '_DSC3115.jpg'
#file_name = 'black200.png'
#file_name = '023.jpg'

#quantization table for Quality factor of 50
Q = np.matrix([[16,11,10,16,24,40,51,61],
               [12,12,14,19,26,58,60,55],
               [14,13,16,24,40,57,69,56],
               [14,17,22,29,51,87,80,62],
               [18,22,37,56,68,109,103,77],
               [24,35,55,64,81,104,113,92],
               [49,64,78,87,103,121,120,101],
               [72,92,95,98,112,100,103,99]],dtype='uint8')


# | FUNCTIONS |
#------------------------------------------------------------------------------
# INITIALIZER returns a NxN matrix of cosines evaluated at k = N discrete
# points. Inverse of this matrix is also returned for use in the inverse DCT.
# Also opens image by file_name in same folder as this program. Converts to
# YCbCr format and turns into an array. Returns ycbcr array.
def Initializer(N):
    K = np.matrix(np.arange(0,N,1)).T
    arg_1 = []
    for n in range(N):
        arg_n = ((((2*n)+1)*pi)/(2*N))
        arg_1.append(arg_n)
    arg_full = np.full((N,N),arg_1)
    arg_k = np.multiply(K,arg_full)
    COS_k = np.cos(arg_k)
    COS_k_inv = np.linalg.inv(COS_k)
    
    image = Image.open(file_name)
    ycbcr = image.convert('YCbCr')
    ycbcr_arr = np.array(ycbcr)
    
    return COS_k, COS_k_inv, ycbcr_arr



#------------------------------------------------------------------------------
# IMAGE_PADDER_BASIC handles the case if an image is not divisible into NxN
# chonks. Does the most basic row and column padding - bottom and right.
def Image_Padder_Basic(N,ycbcr_arr):
    new_rows = input_rows = ycbcr_arr.shape[0]
    new_columns = input_columns = ycbcr_arr.shape[1]
    pad_r = input_rows%N
    pad_c = input_columns%N
    if input_rows%N != 0:
        pad_r_array = np.full((pad_r,input_columns,3),[16,128,128],dtype='uint8')
        new_rows += pad_r
        ycbcr_arr = np.append(ycbcr_arr,pad_r_array,axis=0)
    if input_columns%N != 0:
        pad_c_array = np.full((new_rows,pad_c,3),[16,128,128],dtype='uint8')
        new_columns += pad_c
        ycbcr_arr =  np.append(ycbcr_arr,pad_c_array,axis=1)
    return ycbcr_arr



#------------------------------------------------------------------------------
# GREY_SCALER blanks the Cb and Cr channels to 127 so that only the Y channel
# remains.
def Grey_Scaler(ycbcr_arr):
    cbcr_blank = np.full((ycbcr_arr.shape[0],ycbcr_arr.shape[1],2),[127,127],dtype='uint8')
    if ycbcr_arr.shape[2] == 1:
        y_channel = ycbcr_arr
    elif ycbcr_arr.shape[2] == 3:
        y_channel = ycbcr_arr[:,:,[0]]
    else:
        print("input array does not meet size restrictions")
    greyscale_arr = np.append(y_channel,cbcr_blank,axis=2)
    return greyscale_arr



#------------------------------------------------------------------------------
# ENCODER looks through the channel input and separates it into NxN "chonks".
# Then calls DCT_2D and Quantizer subfunctions to create a NxN coefficient
# matrix for each chonk and writes a coefficient matrix for entire channel.
def Encoder(N,channel_arr,Q,COS_k):
    
    
    # DCT_2D iterates through a NxN sample (chonk) and calls the DCT function
    # to compute the DCT coefficients for the rows and then columns to form a
    # NxN coefficient matrix. 
    def DCT_2D(N,chonk,COS_k):
        
        # DCT computes the DCT coefficiencts for a Nx1 block (chunk).
        # Chunk input is a 1xN matrix and must be transposed first to Nx1.
        def DCT(chunk,COS_k):
            chunk = chunk.T
            Coef_1D = np.dot(COS_k,chunk)
            return Coef_1D.T
        
        
        row_wise_coef = []
        for n in range(N):
            row_n_coef = DCT(chonk[n],COS_k).tolist()
            row_wise_coef = row_wise_coef + row_n_coef
        row_wise_coef = np.matrix(row_wise_coef)

        column_wise_coef = []
        for n in range(N):
            column_n_coef = DCT(row_wise_coef[:,n].T,COS_k).tolist()
            column_wise_coef = column_wise_coef + column_n_coef
        column_wise_coef = np.matrix(column_wise_coef).T
        return column_wise_coef
    
    
    # QUANTIZER divides the 2D DCT coefficients by a JPEG standard quantization
    # matrix to remove higher frequency signals less detectable by human eyes.
    def Quantizer(Coef_2D,Q):
        Quan_Coef = np.divide(Coef_2D,Q)
        Quan_Coef = Quan_Coef.round()
        return Quan_Coef
    
    
    channel_arr = channel_arr - 128
    
    num_row_chonks = int(channel_arr.shape[0]/N)
    num_col_chonks = int(channel_arr.shape[1]/N)
    for chonk_row in range(num_row_chonks):
        for chonk_col in range(num_col_chonks):
            chonk = channel_arr[(chonk_row*N):((chonk_row+1)*N),(chonk_col*N):((chonk_col+1)*N)]
            chonk = np.matrix(chonk)
            chonk_coef = DCT_2D(N, chonk, COS_k)
            chonk_coef = Quantizer(chonk_coef, Q)
            if chonk_col == 0:
                n_row_coef = chonk_coef
            else:
                n_row_coef = np.append(n_row_coef,chonk_coef,axis=1)
        if chonk_row == 0:
            coef_matrix = n_row_coef
        else:
            coef_matrix = np.append(coef_matrix,n_row_coef,axis=0)
        coef_matrix = coef_matrix.astype(np.int16)
    return coef_matrix



#------------------------------------------------------------------------------
# DECODER creates a lossy recreation of the original input image array.
# Calls invDCT_2D and deQuantizer subfunctions for each NxN chonk and writes
# a matrix that recreates a lossy version of the original input array.
def Decoder(N,coef_arr,Q,COS_k_inv):
    
    
    # INVDCT_2D computes the NxN compressed sample (chonk) from DCT coefficients.
    def invDCT_2D(N,Coef_2D,COS_k_inv):
        
        # INVDCT computes the Nx1 block (chunk) from the DCT coefficients.
        # Output is transposed to 1xN for ease of use.
        def invDCT(Coef_1D,COS_k_inv):
            Coef_1D = Coef_1D.T
            chunk = np.dot(COS_k_inv,Coef_1D)
            return chunk.T
            
        
        column_wise_data = []
        for n in range(N):
            column_n_data = invDCT(Coef_2D[:,n].T,COS_k_inv).tolist()
            column_wise_data = column_wise_data + column_n_data
        column_wise_data = np.matrix(column_wise_data).T

        row_wise_data = []
        for n in range(N):
            row_n_data = invDCT(column_wise_data[n],COS_k_inv).tolist()
            row_wise_data = row_wise_data + row_n_data
        row_wise_data = np.matrix(row_wise_data)
        return row_wise_data
    
    
    # DEQUANTIZER multiplies quantized 2D DCT coefficients by quantization matrix
    # to get DCT coefficient matrix ready to be used to compute the output image.
    def deQuantizer(Quan_Coef,Q):
        deQuan_Coef = np.multiply(Quan_Coef,Q)
        return deQuan_Coef
    
    
    num_row_chonks = int(coef_arr.shape[0]/N)
    num_col_chonks = int(coef_arr.shape[1]/N)
    for chonk_row in range(num_row_chonks):
        for chonk_col in range(num_col_chonks):
            chonk = coef_arr[(chonk_row*N):((chonk_row+1)*N),(chonk_col*N):((chonk_col+1)*N)]
            chonk = np.matrix(chonk)
            chonk_coef = deQuantizer(chonk,Q)
            chonk_data = invDCT_2D(N, chonk_coef, COS_k_inv)
            if chonk_col == 0:
                n_row_data = chonk_data
            else:
                n_row_data = np.append(n_row_data,chonk_data,axis=1)
        if chonk_row == 0:
            data_matrix = n_row_data
        else:
            data_matrix = np.append(data_matrix,n_row_data,axis=0)
    data_matrix = data_matrix + 128
    data_matrix = np.array(data_matrix, dtype='uint8')
    return data_matrix
    


# | MAIN |
#------------------------------------------------------------------------------
ck, ck_inv, ycbcr_arr = Initializer(N)

if (ycbcr_arr.shape[0]%N != 0) or (ycbcr_arr.shape[1]%N != 0):
    ycbcr_arr_padded = Image_Padder_Basic(N, ycbcr_arr)
else:
    ycbcr_arr_padded = ycbcr_arr
    
grey_arr = Grey_Scaler(ycbcr_arr_padded)

y_channel = grey_arr[:,:,0]
y_channel = y_channel.astype('int8')
encoded = Encoder(N,y_channel,Q,ck)
decoded = Decoder(N,encoded,Q,ck_inv)

cbcr_blank = np.full((decoded.shape[0],decoded.shape[1],2),[127,127],dtype='uint8')
decoded = decoded[:,:,newaxis]
grey_arr_decoded = np.append(decoded,cbcr_blank,axis=2)
#grey_arr_decoded = Grey_Scaler(decoded)


# | VISUALIZING RESULTS |
#------------------------------------------------------------------------------
#ycbcr_padded_image = Image.fromarray(ycbcr_arr_padded,"YCbCr")
input_greyscale = Image.fromarray(grey_arr,"YCbCr")
output_greyscale = Image.fromarray(grey_arr_decoded,"YCbCr")

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1 = plt.imshow(input_greyscale,aspect='auto')
plt.title("INPUT")
ax2 = fig.add_subplot(122)
ax2 = plt.imshow(output_greyscale,aspect='auto')
plt.title("OUTPUT")
#ax3 = fig.add_subplot(223)
#ax3 = plt.imshow(grey_image,aspect='auto')
#plt.title("GreyScale")
#plt.show()