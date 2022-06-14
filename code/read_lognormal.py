import numpy as np
import struct


def read(fn):

    with open(fn, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        
        nleading = 3*8+1*4
        header = struct.unpack("dddi", fileContent[:nleading])
        Lx, Ly, Lz, N = header
        data = struct.unpack("f" * ((len(fileContent) -nleading) // 4), fileContent[nleading:])
         
    data = np.array(data)
    data = data.reshape((-1, 6)) 
        
    return Lx, Ly, Lz, N, data
