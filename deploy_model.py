# from tools import test_net
from utils import parser
# from utils import dist_utils, misc
from utils.logger import *
from utils.config import *
from tools import builder
import time
import os
import torch
from IPython import embed


def return_model():
	# args
	args = parser.get_args()
	# CUDA
	args.use_gpu = torch.cuda.is_available()
	if args.use_gpu:
		torch.backends.cudnn.benchmark = True
	# init distributed env first, since logger depends on the dist info.
	args.distributed = False
	
	# Creating a logger. 
	timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
	logger = get_root_logger(log_file=log_file, name=args.log_name)

	# Writing tings to config files. 
	config = get_config(args, logger = logger)    
	# batch size
	config.dataset.train.others.bs = config.total_bs
	config.dataset.val.others.bs = 1
	config.dataset.test.others.bs = 1
	
	# Logging things.
	log_args_to_file(args, 'args', logger = logger)
	log_config_to_file(config, 'config', logger = logger)    
	logger.info(f'Distributed training: {args.distributed}')

	# test_net(args, config)
	logger = get_logger(args.log_name)
	print_log('Tester start ... ', logger = logger)
	_, test_dataloader = builder.dataset_builder(args, config.dataset.test)

	base_model = builder.model_builder(config.model).cuda()
	return base_model

def main():
	pointmae_model = return_model()
	print("Embed after model creation.")
	embed()

if __name__ == '__main__':
	main()
