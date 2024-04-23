import argparse
import numpy as np
import scipy.fft as fft


# parser.parse_args('a b --foo x y --bar 1 2'.split())

# TODO: Include command line args 
parser = argparse.ArgumentParser()
# parser.add_argument('-r', '--rank', help="Rank of desired output tensor", type=int, default=2)
# parser.add_argument('dimensions', metavar='N', type=int, nargs=rank,
parser.add_argument('dimensions', metavar='N', type=int, default=[4, 1], nargs='+',
                    help='Dimensions for each rank of desired output tensor (e.g. 4 1 for a 4x1 matrix)')
# Input is of form [[0., 1., 2.], [3., 4., 5.]] to match the string representation of a numpy array so one can "easily" pass the output of printing as input
parser.add_argument('-i', '--input', help="Input string to convert to file", type=str, default='float32')
parser.add_argument('-o', '--output-file', help="Data type of desired output tensor", type=str, default='float32')
data_types = ['float32', 'float64', 'complex64', 'complex128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
parser.add_argument('-t', '--type', help=f"Data type of desired output tensor. Choices: {data_types}", type=str, default='float32', choices=data_types)
parser.add_argument('-b', help="Binary data output", action='store_true')

# parser.add_argument('-c', '--complex', help="Complex data output", action='store_true')

# group = parser.add_mutually_exclusive_group()
# group.add_argument('-b', help="Binary data output", action='store_true')
# group.add_argument('-i', help="Integer data output", action='store_true')
# group.add_argument('-f', help="Floating-point data output", action='store_true')
# group.add_argument('--float32', help="Floating-point data output", action='store_true')
# group.add_argument('--float64', help="Floating-point data output", action='store_true')
args = parser.parse_args()


def datagen(dimensions, data_type, binary=False):
    #TODO: Generate data I guess? would want the arguments to function to roughly match the parser args
    pass

if __name__ == '__main__':
    if args.rank != len(args.dimensions):
        print("Error: Rank does not match number of dimensions")
        exit()
    # Do stuff here