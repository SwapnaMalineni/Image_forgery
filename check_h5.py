# save as check_h5.py and run: python check_h5.py path/to/forgery_model.h5
import sys, os
path = sys.argv[1]
if not os.path.exists(path):
    print('not found', path); sys.exit(1)
with open(path,'rb') as f:
    header = f.read(8)
print('header bytes:', header)
print('looks valid HDF5:' , header == b'\x89HDF\r\n\x1a\n')