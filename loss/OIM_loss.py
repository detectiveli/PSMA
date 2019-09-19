from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

class _OIM_Module(Function):
    def __init__(self, LUT, num_classes):
        super(_OIM_Module, self).__init__()
        self.LUT = LUT
        self.momentum = 0.5  # TODO: use exponentially weighted average
        self.num_classes = num_classes
    # @staticmethod
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.LUT.t())
        return outputs_labeled
    # @staticmethod
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.LUT)
        for feat_each, each_person in zip(inputs, targets): #each person from proposal
            self.LUT[each_person, :] = self.momentum*self.LUT[each_person, :] + (1 - self.momentum)*feat_each
        # print(self.QUEUE[-1])
        return grad_inputs, None

class OIM_Module(nn.Module):
    def __init__(self, class_size):
        super(OIM_Module, self).__init__()
        # self.updated = 0
        self.num_feature = 2048
        self.num_classes = class_size

        self.register_buffer('LUT', torch.zeros(
            self.num_classes, self.num_feature).cuda())
        self.momentum_ = 0.5

    def forward(self, x, person_id):
        labed_score = _OIM_Module(self.LUT, self.num_classes)(x, person_id)
        return labed_score