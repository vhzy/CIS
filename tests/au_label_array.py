import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-a',
                        '--processor_name',
                        type=int,
                        help='processor name')
parser.add_argument('-b',
                        '-config_dir',
                        type=int,
                        help='config dir name')

args = parser.parse_args()
print(type(args))
print(sys.argv[2:])