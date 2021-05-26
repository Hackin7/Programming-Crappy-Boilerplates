#!/usr/bin/env python3

xor_vals = [42, 146, 111, 49, 198, 242, 39, 46, 186, 169, 194, 116, 171, 154, 42, 211, 120, 98, 220, 142, 76, 12, 252, 67, 183, 106]
enc_flag = [105, 203, 60, 74, 180, 194, 95, 26, 212, 199, 241, 43, 211, 170, 88, 140, 10, 82, 191, 229, 19, 57, 200, 45, 211, 23]
flag = ""
for flag_byte,xor_byte in zip(xor_vals,enc_flag):
    flag += chr(flag_byte ^ xor_byte)
print(flag)
