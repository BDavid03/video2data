"""CPU F1 queue entry point."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['stage-a'])
    parser.add_argument('--request', default='src/features/mesh_fusion/queue.json')
    args = parser.parse_args()
    from .loop import run_queue
    run_queue(args.request)


if __name__ == '__main__':
    main()
