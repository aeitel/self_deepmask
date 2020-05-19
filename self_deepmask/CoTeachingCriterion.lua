-- Copyright (c) 2020,Andreas Eitel, All rights reserved.


require 'nn'

local CoTeachingCriterion = torch.class('nn.CoTeachingCriterion')

function CoTeachingCriterion:__init(config)
  self.criterion = nn.SoftMarginCriterion():cuda()
  self.criterion2 = nn.SoftMarginCriterion():cuda()
  print("Co-teaching Learning Loss")
  local ind1_update
  local ind2_update
end

function CoTeachingCriterion:forward(input1,input2,target,forget_rate)
  local loss1_tensor = torch.DoubleTensor(input1:size(1),1):zero()
  local loss2_tensor = torch.DoubleTensor(input2:size(1),1):zero()
  for i=1,input1:size(1) do
    local loss1 = self.criterion:forward(input1[i], target[i])
    local loss2 = self.criterion2:forward(input2[i], target[i])
    loss1_tensor[i] = loss1
    loss2_tensor[i] = loss2
  end

  local loss1_sorted, ind1_sorted = torch.sort(loss1_tensor,1)
  local loss2_sorted, ind2_sorted = torch.sort(loss2_tensor,1)
  local remember_rate = 1 - forget_rate
  local num_remember = remember_rate * loss1_sorted:size(1)
  ind1_update=ind1_sorted[{{1,num_remember},1}]
  ind2_update=ind2_sorted[{{1,num_remember},1}]
  -- exchange
  local loss1_update = self.criterion:forward(input1:index(1,ind2_update), target:index(1,ind2_update))
  local loss2_update = self.criterion2:forward(input2:index(1,ind1_update), target:index(1,ind1_update))
  --print("Loss1 tensor mean",loss1_update,"loss2 tensor mean",loss2_update)
  return loss1_update,loss2_update
end

function CoTeachingCriterion:backward(input1,input2, target)
  --print("input1",input1,"input2",input2)
  local gradOutputs1 = self.criterion:backward(input1, target)
  local gradOutputs2 = self.criterion2:backward(input2, target)
  local mask1 = torch.ByteTensor():resize(input1:size(1),1):fill(1)
  local mask2 = torch.ByteTensor():resize(input2:size(1),1):fill(1)
  -- exchange
  for i=1,ind1_update:size(1) do
    mask1[ind2_update[i]][1] = 0
    mask2[ind1_update[i]][1] = 0
  end
  mask1 = mask1:cudaByte()
  mask2 = mask2:cudaByte()
  gradOutputs1:maskedFill(mask1, 0.0)
  gradOutputs2:maskedFill(mask2, 0.0)
  return gradOutputs1, gradOutputs2
end
