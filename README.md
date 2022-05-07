# Python-JPG-Encoder
 Python implementation of a JPG encoder.
As of 5/7/22:
1. Takes in an image (PNG, JPEG) and converts from RGB to YCbCr colorspace
2. Pads image with black border if not divisible by 8x8 pixel chunks
3. Extracts the Lumens channel only
4. Performs 2-dimensional Direct Cosine transform on 8x8 pixel chunks to produce a matrix coefficient matrix
5. Divides by a Quality Factor = 50 JPEG Quantization table to remove high frequency signals not easily noticeable by the human eye
6. "Decoder" reverses all the previous effets to reproduce a greyscale image with slightly less quality


Actual encoding to a bin file is not implemented at this time. This is only a proof of concept for the 2-D DCT