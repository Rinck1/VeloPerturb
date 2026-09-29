import argparse
import json
from veloroute.gfg_resume import resume_gfg

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-directory', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
print(json.dumps(resume_gfg(args.checkpoint_directory, args.output, device=args.device)))
