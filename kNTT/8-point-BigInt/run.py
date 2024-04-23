#!/usr/bin/env cs_python

import struct
import math
import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module


def print_hex(data: np.ndarray, indices: list = None):
    print(" ".join([f"{x:16x}" for x in data]))

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


dtype = np.uint32
# dtype = np.ulong

input_vector = np.array(
                          [66180002, 46054340, 185133904, 68728779, 156137740, 135212474, 87317267, 204961903], 
                          dtype=dtype
                        ).reshape(1, 8)  # Make it a row vector
expected_output = np.array(
                            [177566360, 54723412, 49777885, 162151521, 82033502, 102309405, 212936481, 224812172], 
                            dtype=dtype
                          ).reshape(1, 8)  # Make it a row vector
print(f"Expected output = {expected_output}")
print("input vector shape: ", input_vector.shape)

# Variables
width = 8
M = 8
N = 8

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
# TODO: Is there a MEMCPY_64BIT? Update: Currently no MEMCPY_64BIT
# y_result = np.zeros([2*2*4*width], dtype=dtype)
y_result = np.zeros([2*width*width], dtype=dtype)  # Times 2 to account for 2 32-bit ints to make 64-bit int

# runner.memcpy_d2h(dest=y_result, src=y_symbol, px=0, py=7, w=1, h=1, elem_per_pe=8, streaming=False,
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 8, 1, 16, streaming=False,
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
print(f"y_result = {y_result.reshape(8, 8)}")

print("SUCCESS? ")