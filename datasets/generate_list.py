import os
import math
import argparse
import numpy as np

IMAGENET_DIR = None
DIR_LIST = ['/home/user/ILSVRC2012/',
			'/home/real/ILSVRC2012/',
			'/data/ILSVRC2012/']

for path in DIR_LIST:
	if os.path.exists(path):
		IMAGENET_DIR = path
		break

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root_dir', type=str)
	parser.add_argument('--percent', type=float, default=0.2)
	parser.add_argument('--outfile', type=str, default='datasets/lists/20.txt')
	args = parser.parse_args()
	train_path = os.path.join(args.root_dir, 'train')
	lines = []
	for name in os.listdir(train_path):
		samples = os.listdir(os.path.join(train_path, name))
		N_c = len(samples)
		sample_N = int(math.ceil(N_c * args.percent))
		sub_samples = np.random.choice(samples, sample_N, replace=False)
		print('Sample {} samples from {}'.format(sample_N, name))
		for s in sub_samples:
			lines.append('{},{}\n'.format(os.path.join(name, s), name))
	with open(args.outfile, 'w') as f:
		f.writelines(lines)

main()