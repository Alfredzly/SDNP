import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

def refine_p_label(output, proto, output_h, proto_h):
    proto = F.interpolate(proto, size=(224, 224), mode='nearest')

    proto_h = F.interpolate(proto_h, size=(224, 224), mode='nearest')
    a = []
    b = []
    for i in range(output.shape[0]):
        output_sample = output[i:i+1]
        output_h_sample = output_h[i:i+1]
        proto_sample = proto[i:i+1]
        proto_h_sample = proto_h[i:i+1]

        pred = torch.argmax(output_sample, dim=1)
        pred_p = torch.argmax(output_h_sample, dim=1)
        pred_h = torch.argmax(proto_sample, dim=1)
        pred_p_h = torch.argmax(proto_h_sample, dim=1)

        disagree = pred!=pred_h
        disagree_p = pred_p!=pred_p_h

        C_fcn_o = -torch.sum(output_sample * torch.log(output_sample + 1e-8), dim=1)
        C_proto_o = -torch.sum(proto_sample * torch.log(proto_sample + 1e-8), dim=1)
        C_proto_a = -torch.sum(proto_h_sample * torch.log(proto_h_sample + 1e-8), dim=1)
        C_fcn_a = -torch.sum(output_h_sample * torch.log(output_h_sample + 1e-8), dim=1)
        C_f = C_fcn_a[disagree]
        C_f_1 = C_fcn_o[disagree]
        C_p = C_proto_a[disagree_p]
        C_p_1 = C_proto_o[disagree_p]
        eps = 1e-8
        a.append(C_f_1.mean() / (C_f.mean() + eps))
        b.append(C_p_1.mean() / (C_p.mean() + eps))
        # print(a, b)
        
    return a, b

def get_label(output, proto, output_h, a=[], b=[], th=1.0):
    proto = F.interpolate(proto, size=(224, 224), mode='nearest')
    # proto_h = F.interpolate(proto_h, size=(256, 256), mode='nearest')
    pred_list = []
    output_list = []

    for i in range(output_h.shape[0]):
        output_sample = output[i:i+1]
        output_h_sample = output_h[i:i+1]
        proto_sample = proto[i:i+1]
        if a[i] > 0.8 and b[i] > 0.8:
            output_final = (output_h_sample * a[i] + proto_sample * b[i])
        else:
            output_final = (output_sample + proto_sample)
        pred_final = torch.argmax(output_final, dim=1)
        pred_list.append(pred_final)
        output_list.append(output_final)
    pred_final = torch.cat(pred_list, dim=0)
    output_final = torch.cat(output_list, dim=0)
    return output_final, pred_final