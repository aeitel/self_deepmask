-- Copyright (c) 2020,Andreas Eitel, All rights reserved.
require 'nn'

local SPLCriterion = torch.class('nn.SPLCriterion')

function SPLCriterion:__init(config)
  self.criterion = nn.SoftMarginCriterion():cuda()
  self.threshold = 0.002 --average loss in first epoch without filtering
  self.growing_factor = 1.2
  self.batch_size = config.batch
  print("Self-paced Learning Loss initialized with threshold",self.threshold,"growing factor",self.growing_factor)
  self.v = torch.IntTensor(self.batch_size,1):zero()
end

function SPLCriterion:increase_threshold()
  self.threshold = self.threshold * self.growing_factor
end

function SPLCriterion:forward(input, target)
  local loss_tensor = torch.DoubleTensor(input:size(1),1):zero()
  local number_hard = 0
  for i=1,input:size(1) do
    local loss = self.criterion:forward(input[i], target[i])
    if loss < self.threshold then
      loss_tensor[i] = loss
      self.v[i] = 1
    else 
      self.v[i] = 0
      number_hard = number_hard + 1
    end
  end
  --print("Loss tensor mean",torch.mean(loss_tensor),"current threshold",self.threshold)
  --print("Removed hard examples",number_hard)
  return torch.mean(loss_tensor)

end

function SPLCriterion:backward(input, target)
  local gradOutputs = self.criterion:backward(input, target)
  for i=1,input:size(1) do
    local vi = self.v[i][1]
    if vi == 0 then
      gradOutputs[i] = 0
    end
  end
  return gradOutputs
end
