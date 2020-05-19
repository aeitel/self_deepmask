-- Copyright (c) 2020,Andreas Eitel, All rights reserved.
require 'nn'

local ReedHardCriterion = torch.class('nn.ReedHardCriterion')
--This loss function is proposed in:
--Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping", 2014

function ReedHardCriterion:__init(config)
  self.criterion = nn.SoftMarginCriterion():cuda()
  self.batch_size = config.batch
  self.beta = 0.7
  print("ReedHard Learning Loss beta",self.beta)
  self.v = torch.IntTensor(self.batch_size,1):zero()
  local target_tanh
end

function ReedHardCriterion:forward(input, target)
  -- Convert all to probabilities
  local target_prob = torch.gt(target,0):double()
  -- we use hard prediction targets, MAP estimate
  local pred_prob = torch.gt(input,0):double()
  local numerical_offset = 0.00000000001
  local target_true_update =  self.beta * target_prob + (1 - self.beta) * pred_prob - (target:double()*numerical_offset) -- we use an offset for numerical stability
  -- convert probabilities to logits
  local target_logit =torch.log(torch.cdiv(target_true_update,(-target_true_update+1)))
  -- map logits to -1 and 1, as expected by SoftMarginCriterion
  target_tanh = torch.tanh(target_logit)
  local loss_batch = self.criterion:forward(input,target_tanh:cuda())
  return loss_batch
end

function ReedHardCriterion:backward(input, target)
  local gradOutputs = self.criterion:backward(input,target_tanh:cuda())
  return gradOutputs
end
