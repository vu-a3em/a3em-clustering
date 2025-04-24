import torch
import torch.quantization
import onnx
import mfcc_vae_4 as mfcc_vae
import dataloader
import mfcc
import onnx_tf
import tensorflow as tf
import numpy as np
import random

ONNX_MODEL_PATH = 'model.onnx'
TF_MODEL_PATH = 'model.pb'
TFLITE_MODEL_PATH = 'model.tflite'
ONNX_OPSET_VERSION = 13

QAT_EPOCHS = 5
BATCH_SIZE = 35
LR = 1e-4

# load the pytorch model

encoder = mfcc_vae.Encoder(embedding_size = 16)
# encoder.load_state_dict(torch.load('mfcc-untested-3/encoder-F16-A0.999-E256-L183.pt'))
encoder.load_state_dict(torch.load('mfcc-4-untested-3/encoder-F16-A0.95-E256-L36.pt'))
encoder.eval()
encoder.cpu()

# perform quantization-aware training

print('loading dataset...')
raw_dataset = dataloader.get_dataset(None, 8192)

print('prepping dataset...')
prepared = []
with torch.no_grad():
    for label, samples in raw_dataset.items():
        for sample in samples:
            x = torch.tensor(mfcc.mfcc_spectrogram_for_learning(sample, dataloader.UNIFORM_SAMPLE_RATE).astype(np.float32))
            y = encoder.forward(x.reshape(1, *x.shape))[0][0].detach()
            prepared.append((x, y))

encoder.train()
encoder.fuse_model()
encoder.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
encoder = torch.quantization.prepare_qat(encoder)

opt = torch.optim.AdamW(encoder.parameters(), lr = LR, amsgrad = True)

print('\nqat training...')
for epoch in range(QAT_EPOCHS):
    random.shuffle(prepared)
    total_loss = 0
    for i in range(len(prepared) // BATCH_SIZE):
        src = prepared[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        input = torch.stack([v[0] for v in src]).detach()
        expected = torch.stack([v[1] for v in src]).detach()
        actual = encoder.forward(input)[0]

        opt.zero_grad()
        loss = torch.mean(torch.sum((actual - expected)**2, dim = 1), dim = 0)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        print(f'\repoch {epoch + 1}/{QAT_EPOCHS} ({100 * (i + 1) / (len(prepared) // BATCH_SIZE):.2f}%) : avg mse {total_loss / (i + 1):.4f})                                            ', end = '')
    print()
print('qat training complete...\n')

encoder.eval()

# convert pytorch model to onnx

qat_modules = (
    torch.quantization.FakeQuantize,
    torch.quantization.MinMaxObserver,
    torch.quantization.MovingAverageMinMaxObserver,
    torch.quantization.PerChannelMinMaxObserver,
    torch.quantization.MovingAveragePerChannelMinMaxObserver,
)
def remove_qat_modules(model):
    for name, module in model.named_children():
        if isinstance(module, qat_modules):
            setattr(model, name, torch.nn.Identity())
        else:
            remove_qat_modules(module)
    return model
encoder = remove_qat_modules(encoder) # not ideal but onnx doesn't support quant stubs

torch.onnx.export(encoder, torch.rand(1, 16, 65), ONNX_MODEL_PATH, opset_version = ONNX_OPSET_VERSION)

# map some symbol names because onnx is buggy apparently
# https://stackoverflow.com/questions/76839366/tf-rep-export-graphtf-model-path-keyerror-input-1

onnx_model = onnx.load(ONNX_MODEL_PATH)

name_map = {
    'onnx::Reshape_0': 'onnx_reshape_0',
    'x.1': 'x_1',
}

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

print('processing dataset for quantization metrics...')
def preprocess_stream():
    for x, y in prepared:
        yield [x]

converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = preprocess_stream
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print('success!')
