from model import Encoder, LatentVariation, Decoder
import torch
import numpy as np
import matplotlib.pyplot as plt


def paddedInputsFromVocab(Vocab, padding=0):
    sent_emb = [[float(i) for i,x in enumerate(s.split())] for s in Vocab.split('.')]
    sent_emb = np.array(sent_emb)
    max_len_v = max([len(v) for v in sent_emb])
    assert padding in [0,1,-1], 'must be between -1, 1'
    if padding is 0:
        padding = torch.zeros(sent_emb.shape[0], max_len_v)
    elif padding == 1:
        padding = torch.ones(sent_emb.shape[0], max_len_v)
    elif padding == -1:
        pass
    

    for v_i in range(padding.shape[0] - 1):
         padding[(v_i), :len(sent_emb[v_i])] = torch.Tensor(sent_emb[v_i])
    
    input_s, spn = padding.shape

    return padding, input_s, spn

def insertInEncoder(params, vocab, inter_rng = 2):        
    inputs, input_s, spn = params
    inputs = np.expand_dims(inputs, axis=2).T
    inputs = torch.Tensor(inputs)
    encoder = Encoder(input_s, spn)
    out, hd = encoder.forward(inputs)
    x,y,a  = out.shape
    out_l = Encoder(x,y)
    out  =  torch.Tensor(out.T.shape)
    out, hd =  out_l.forward(out)
    for i in range(inter_rng):
        out_l = Encoder(out.shape[-1],spn)
        out, hd = out_l.forward(out)

    return out

if __name__ == '__main__':

    vocab = """
     I will feel more alive
    """
    inputs = paddedInputsFromVocab(vocab, padding=1)

    out = insertInEncoder(inputs, vocab, inter_rng=6)
    axes = [out[:,:,0], out[0,:,:], out[:,0,:]]
    ax, ay,az = [],[],[]
    conts =  [ax,ay,az]
    for dim, cont in zip(axes,conts):
        axis = []
        for sl in (out[:,:,0]):
            cont.append(sl.detach().numpy())
    
        print(torch.Tensor(cont).shape)

    x = (np.array(ax))

    plt.plot(x)
    plt.show()
    #int_Tensor = out.type(torch.int32)
    #out_dense = out.reshape(out.shape[1:])
