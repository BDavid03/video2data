"""Thin command dispatcher for the frozen pilot."""

import argparse
from .pilot import PilotRequest, pilot, smoke

parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=('pilot','smoke','g2','g3'))
parser.add_argument('--request', default='track3/features/mesh_generative/request.json')
args = parser.parse_args()
if args.operation == 'g3':
    from .g3 import run
    print(run())
elif args.operation == 'g2':
    from .g2 import run
    print(run())
else:
    print(smoke() if args.operation == 'smoke' else pilot(PilotRequest.from_file(args.request)))
