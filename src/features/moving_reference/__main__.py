"""Run M1 masks, the sealed CPU tracker, or freeze candidate A before episode 38."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', nargs='?', choices=['masks', 'track', 'freeze'], default='masks')
    parser.add_argument('--episode', type=int, choices=[39, 38], help='Resume one authorized queue item')
    parser.add_argument('--attempt', default='candidate-a')
    args=parser.parse_args()
    if args.operation=='masks':
        from .masks import run
        run(args.episode)
    elif args.operation=='freeze':
        from .track import freeze
        freeze()
    else:
        if args.episode is None:
            parser.error('track requires --episode')
        from .track import run
        run(args.episode,args.attempt)
