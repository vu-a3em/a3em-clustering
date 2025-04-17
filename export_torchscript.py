import torch
import mfcc_vae_1 as mfcc_vae
import dataloader
import mfcc

encoder = mfcc_vae.Encoder(embedding_size = 16)
encoder.load_state_dict(torch.load('mfcc-untested-3/encoder-F16-A0.999-E256-L183.pt'))
encoder.eval()

# -- normal --

example = torch.rand(1, 65, 65)
torch.jit.trace(encoder, example).save('portable-model-f32.pt')

# -- quantized --

encoder.qconfig = torch.quantization.get_default_qconfig('qnnpack')
encoder_prep = torch.quantization.prepare(encoder, inplace = False)

raw_dataset = dataloader.get_dataset(None, 8192)
for cls, samples in raw_dataset.items():
    for sample in samples:
        encoder_prep(torch.tensor(mfcc.mfcc_spectrogram_for_learning(sample, dataloader.UNIFORM_SAMPLE_RATE), dtype = torch.float32).reshape(1, 65, 65))

encoder_quant = torch.quantization.convert(encoder_prep, inplace = False)
print(encoder_quant)
torch.jit.trace(encoder_quant.to('cpu'), example).save('portable-model-i8.pt')

print('success!')
