# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:06:36 2020
fixes tensorflow-gpu memory allocation crashes
configures GPU to use memory as its needed instead of pre-allocating
updated for Tensorflow 2
@author: Adam Santos
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
