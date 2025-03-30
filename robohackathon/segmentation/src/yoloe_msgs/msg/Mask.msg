# Mask.msg
# This message represents a 2D mask as a flattened boolean array.
# The data is stored in row-major order.

uint32 rows   # Number of rows in the mask
uint32 cols   # Number of columns in the mask
bool[] data   # Flattened boolean array (length should equal rows * cols)
