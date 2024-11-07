import torch
import onnx
import mfcc_vae_1 as mfcc_vae
import dataloader
import mfcc
import onnx_tf
import tensorflow as tf
import numpy as np

ONNX_MODEL_PATH = 'model.onnx'
TF_MODEL_PATH = 'model.pb'
TFLITE_MODEL_PATH = 'model.tflite'
ONNX_OPSET_VERSION = 13

# load the pytorch model

encoder = mfcc_vae.Encoder(embedding_size = 16)
encoder.load_state_dict(torch.load('mfcc-untested-3/encoder-F16-A0.999-E256-L183.pt'))
encoder.eval()
encoder.cpu()

# convert pytorch model to onnx

example = torch.rand(1, 65, 65)
torch.onnx.export(encoder, example, ONNX_MODEL_PATH, opset_version = ONNX_OPSET_VERSION)

# map some symbol names because onnx is buggy apparently
# https://stackoverflow.com/questions/76839366/tf-rep-export-graphtf-model-path-keyerror-input-1

onnx_model = onnx.load(ONNX_MODEL_PATH)

name_map = { 'onnx::Reshape_0': 'onnx_reshape_0' }

new_inputs = []
for inp in onnx_model.graph.input:
    if inp.name in name_map:
        new_inp = onnx.helper.make_tensor_value_info(name_map[inp.name], inp.type.tensor_type.elem_type, [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
        new_inputs.append(new_inp)
    else:
        new_inputs.append(inp)

onnx_model.graph.ClearField("input")
onnx_model.graph.input.extend(new_inputs)

for node in onnx_model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in name_map:
            node.input[i] = name_map[input_name]

onnx.save(onnx_model, ONNX_MODEL_PATH)

# convert onnx model to tensorflow

onnx_model = onnx.load(ONNX_MODEL_PATH)
onnx_tf.backend.prepare(onnx_model).export_graph(TF_MODEL_PATH)

# convert tensorflow model to tf-lite

def prep_dataset(data):
    X = []
    Y = []
    for i, (label, samples) in enumerate(data.items()):
        for sample in samples:
            X.append(mfcc.mfcc_spectrogram_for_learning(sample, dataloader.UNIFORM_SAMPLE_RATE))
            Y.append(i)
    return np.array(X, dtype = np.float32), np.array(Y, dtype = np.float32)

print('loading dataset...')
raw_dataset = dataloader.get_dataset(None, 8192)

print('processing dataset for quantization metrics...')
def preprocess_stream(raw):
    for label, samples in raw.items():
        for sample in samples:
            yield [mfcc.mfcc_spectrogram_for_learning(sample, dataloader.UNIFORM_SAMPLE_RATE).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = lambda: preprocess_stream(raw_dataset)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
