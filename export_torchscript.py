import torch
import mfcc_vae_1 as vae

encoder = vae.Encoder(embedding_size = 16)
encoder.load_state_dict(torch.load('mfcc-untested-3/encoder-F16-A0.999-E256-L183.pt'))
encoder.eval()

traced = torch.jit.trace(encoder, torch.rand(1, 65, 65))
traced.save('portable-model.pt')
