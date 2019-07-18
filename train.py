from argparse import ArgumentParser
from model import DPE
import tensorflow as tf

"""
python train.py 
"""
tf.enable_eager_execution()
tfe = tf.contrib.eager

parser = ArgumentParser()
parser.add_argument()

model = DPE(1365, 2048)