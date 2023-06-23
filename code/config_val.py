import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--input_data_path', type=str, default='../test',  help='input image path')
parser.add_argument('--ori_data_path', type=str, default='',  help='Origin image path')
parser.add_argument('--gt', type=bool, default=False,  help='gt')
parser.add_argument('--output_dir', type=str, default='../submit_result',  help='output dir')
parser.add_argument('--name', type=str, default='test',  help='name of current run')
parser.add_argument('--type', type=str, default='hdfnet',  help='Model_type')
parser.add_argument('--model_path', type=str, default='../model/weight.pkl',  help='pretrained model path')


def get_config():
	config, unparsed = parser.parse_known_args()
	return config, unparsed
