#!/usr/bin/env cs_python

import struct
import math
import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module


''' Timing helper functions '''
def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])


parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
name = args.name
cmaddr = args.cmaddr

# Parse the compile metadata
with open(f"{name}/out.json", encoding="utf-8") as json_file:
    compile_data = json.load(json_file)
params = compile_data["params"]


input_vector = np.array([1, 2, 3, 4], dtype=np.float32).reshape(1, 4)  # Make it a row vector
expected_output = np.fft.fft(input_vector)
print(f"Expected output = {expected_output}")
print("input vector shape: ", input_vector.shape)

# Variables
width = 4
M = 4
N = 4

# Instantiate runner
runner = SdkRuntime(name, cmaddr=cmaddr)

# Device symbols for memcpy
x_symbol = runner.get_id("x")
y_symbol = runner.get_id("y")
symbol_maxmin_time = runner.get_id("maxmin_time")  # For timing

# Load and begin run
runner.load()
runner.run()


# Launch the main function on device
runner.launch('main', nonblock=False)


# TODO: Need to figure out what the error is here
# Copy input vector to device
# runner.memcpy_h2d(x_symbol, np.tile(input_vector, width), 0, 0, width, 1, N, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


# Copy y back from device
# TODO: What shape should y_result be for complex output?
y_result = np.zeros([2*4*width], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, width, 1, 2*M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# y_result = np.zeros([M], dtype=np.float32)
# runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


# Copy back timestamps from device
# height = 1
# data = np.zeros((width*height*3, 1), dtype=np.uint32)
# runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, width, height, 3,
# streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# maxmin_time_hwl = data.view(np.float32).reshape((height, width, 3))
# print("Copied back timestamps.")

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
# y_result = y_result.reshape(width)
print(f"y_result = {y_result}")

print("SUCCESS? ")




#######################
# Calculate cycle count
#######################
'''
print("Calculating cycle count...")
tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
min_cycles = math.inf
max_cycles = 0

for w in range(width):
    for h in range(height):
        hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 0)]), base=16)
        hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 1)]), base=16)
        hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
        tsc_tensor_d2h[0] = hex_t0 & 0x0000ffff
        tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000ffff
        tsc_tensor_d2h[2] = hex_t1 & 0x0000ffff
        tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000ffff
        tsc_tensor_d2h[4] = hex_t2 & 0x0000ffff
        tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000ffff

    cycles = sub_ts(tsc_tensor_d2h)
    if cycles < min_cycles:
        min_cycles = cycles
        min_w = w
        min_h = h
    if cycles > max_cycles:
        max_cycles = cycles
        max_w = w
        max_h = h
'''