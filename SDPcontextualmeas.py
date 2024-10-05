"""
@author: Yujie Zhang
"""
'''This is a test on certifying and quantifying contextual measurement'''

import Bases
import Utility

M = Bases.Planor(4)   # different measurements are given in Bases.py
Fa, eta = Utility.jrdual(M)    # dual SDP for quantifying white-noise robustness of contextual measurement
#Fa, eta = Utility.jwdual(M)    # dual SDP for quantifying weight of contextual measurement
print("Dual variable (joint_M):", Fa)
print("Critical visibility (eta_jm):", eta)
