import os
from utils import imageE_preprocessing,imageI_preprocessing
from trainer import SUP_for_emb,SUP_for_ion
import argparse
import cv2

parser = argparse.ArgumentParser(
    description='DeepFERE for high-resolution reconstruction of MSI incorporated with mutimodal fusion')

parser.add_argument('--input_MSIEfile',required= True,help = 'path to inputting LR MSI embedding data')
parser.add_argument('--input_MSIIfile',help = 'path to inputting LR MSI single ion data')
parser.add_argument('--input_HEfile',required= True,help = 'path to inputting HR H&E image')
parser.add_argument('--input_shape',required= True,type = int, nargs = '+', help='inputting LR MSI file shape')
parser.add_argument('--n_factor',required=True,type = int, help='user-defined magnification')
parser.add_argument('--mode',
                    help = 'embedding mode for embedding data, ion mode for single ion image',
                    default= 'embedding')
parser.add_argument('--save_result', default=True, type=bool, help='save result for each epoch')
parser.add_argument('--output_file', default='output/',help='output file name')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()
    m,n = args.input_shape

    if args.mode == 'embedding':

        imgg, imgg2, imgg3 = imageE_preprocessing(args.input_MSIEfile,args.input_HEfile,args.n_factor,m,n)

        SUP_for_emb(imgg, imgg2, imgg3,args.n_factor,args.save_result,args.output_file)

    if args.mode == 'ion':

        imgg,imgg2,imgg3,imgg4,imgg5 = imageI_preprocessing(args.input_MSIIfile,args.input_HEfile,args.input_MSIEfile,args.n_factor,m,n)

        SUP_for_ion(imgg,imgg2,imgg3,imgg4,imgg5,args.n_factor,args.save_result,args.output_file)

