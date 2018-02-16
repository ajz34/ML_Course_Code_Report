#---- file util.py
#       All utilities here. These small functions can be useful for
#       other programs of different purposes.
#----

#---- function str2intflt
#       convert string to integer first, then try float;
#       if failed in converting, return the original string.
#----
def str2intflt(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

#---- function ret_index_1D
#       Return the index of the number in the 1D-np.array
#   ! Note:
#   !  1. the number in the list should be and only be occur for one time.
#----
def ret_index_1D(num, lst):
    return int(np.where( lst == num )[0])

