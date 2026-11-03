"""Thin command dispatcher for the frozen pilot."""

import argparse
from .pilot import PilotRequest, pilot, smoke

parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=('pilot','smoke'))
parser.add_argument('--request', default='track3/features/mesh_generative/request.json')
args = parser.parse_args()
print(smoke() if args.operation == 'smoke' else pilot(PilotRequest.from_file(args.request)))
