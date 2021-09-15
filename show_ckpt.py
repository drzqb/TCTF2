"""
    显示ckpt文件中所有变量
    tf1.x环境
"""

from tensorflow.python import pywrap_tensorflow
checkpoint_path = "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names