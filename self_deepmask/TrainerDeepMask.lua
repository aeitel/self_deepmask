--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training and testing loop for DeepMask
------------------------------------------------------------------------------]]
local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--require 'TrainPlotter'
require 'randomkit'
require 'image'

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, model2, criterion, config)
  -- training params
  self.config = config
  self.model = model
  self.maskNet = nn.Sequential():add(model.trunk):add(model.maskBranch)
  self.scoreNet = nn.Sequential():add(model.trunk):add(model.scoreBranch)

  self.criterion = criterion
  self.lr = config.lr
  self.optimState ={}
  for k,v in pairs({'trunk','mask','score'}) do
    self.optimState[v] = {
      learningRate = config.lr,
      learningRateDecay = 0,
      momentum = config.momentum,
      dampening = 0,
      weightDecay = config.wd,
    }
  end
  self.hfreq = config.hfreq

  -- params and gradparams
  self.pt,self.gt = model.trunk:getParameters()
  self.pm,self.gm = model.maskBranch:getParameters()
  self.ps,self.gs = model.scoreBranch:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

  -- log
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
  --self.plotter = TrainPlotter.new(self.rundir .. '/out.json')

  if config.noisefilter == 'coteaching' then
   self.model2 = model2
   self.scoreNet2 = nn.Sequential():add(model2.trunk):add(model2.scoreBranch)
   self.pt2,self.gt2 = model2.trunk:getParameters()
   self.pm2,self.gm2 = model2.maskBranch:getParameters()
   self.ps2,self.gs2 = model2.scoreBranch:getParameters()
   self.modelsv2 = {model2=model2:clone('weight', 'bias'),config=config}
  end


end

function Trainer:trainCoTeaching(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()


  local timer = torch.Timer()


  local fevaltrunk = function() return self.model.trunk.output, self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  local fevalscore = function() return self.criterion.output,   self.gs end

  local fevaltrunk2 = function() return self.model2.trunk.output, self.gt2 end
  local fevalmask2  = function() return self.criterion.output2,   self.gm2 end
  local fevalscore2 = function() return self.criterion.output2,   self.gs2 end

  local moving_loss_average = 0.0
  local iteration = 1
  local forget_rate = self.config.coteachforgetrate
  rate_schedule = torch.ones(self.config.maxepoch)*forget_rate
  print("rate schedule",rate_schedule)

  for n, sample in dataloader:run() do
    -- copy samples to the GPU
    self:copySamples(sample)

    -- forward/backward
    local model, params, feval, optimState
    local model2, params2, feval2, optimState2
    if sample.head == 1 then
      model, params = self.maskNet, self.pm
      feval,optimState = fevalmask, self.optimState.mask
    else
      model, params = self.scoreNet, self.ps
      model2, params2 = self.scoreNet2, self.ps2
      feval,optimState = fevalscore, self.optimState.score
      feval2,optimState2 = fevalscore2, self.optimState.score
    end

    local lossbatch, outputs1, outputs2
    outputs1 = model:forward(self.inputs)
    outputs2 = model2:forward(self.inputs)
    lossbatch = self.criterion:forward(outputs1,outputs2, self.labels,rate_schedule[epoch])
    moving_loss_average = moving_loss_average + lossbatch
    iteration = iteration + 1

    model:zeroGradParameters()
    model2:zeroGradParameters()
    local gradOutputs1, gradOutputs2 = self.criterion:backward(outputs1,outputs2, self.labels)

    if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
    model:backward(self.inputs, gradOutputs1)
    model2:backward(self.inputs, gradOutputs2)

    -- optimize
    optim.sgd(fevaltrunk, self.pt, self.optimState.trunk)
    optim.sgd(fevaltrunk2, self.pt2, self.optimState.trunk)
    optim.sgd(feval, params, optimState)
    optim.sgd(feval2, params2, optimState)

    -- update loss
    self.lossmeter:add(lossbatch)
  end
  print("Epoch",epoch,"Rate schedule",rate_schedule[epoch])
  print("Iterations",iteration,"data loader size",dataloader:size())
  print("Sum loss",moving_loss_average)
  print("Avg loss", moving_loss_average/iteration)

  if config.noisefilter == 'self-paced' then
    self.criterion:increase_threshold()
  end
  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  --torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  print("Epoch ", epoch, epoch%self.config.maxepoch)
  if epoch%self.config.maxepoch == 0 then
    --torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
    --  self.modelsv)
    torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  end

  --self.plotter:info({created_time=io.popen('date'):read(),
  --              tag='My First Plot'})
  --self.plotter:add('Loss', 'training', epoch, self.lossmeter:value())
  collectgarbage()
end

function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()


  local timer = torch.Timer()


  local fevaltrunk = function() return self.model.trunk.output, self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  local fevalscore = function() return self.criterion.output,   self.gs end

  local moving_loss_average = 0.0
  local iteration = 1

  for n, sample in dataloader:run() do
    -- copy samples to the GPU
    self:copySamples(sample)

    -- forward/backward
    local model, params, feval, optimState
    if sample.head == 1 then
      model, params = self.maskNet, self.pm
      feval,optimState = fevalmask, self.optimState.mask
    else
      model, params = self.scoreNet, self.ps
      feval,optimState = fevalscore, self.optimState.score
    end

    local lossbatch, outputs1, outputs2
    outputs = model:forward(self.inputs)
    lossbatch = self.criterion:forward(outputs, self.labels)
    moving_loss_average = moving_loss_average + lossbatch
    iteration = iteration + 1

    model:zeroGradParameters()
    local gradOutputs = self.criterion:backward(outputs, self.labels)

    if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
    model:backward(self.inputs, gradOutputs)

    -- optimize
    optim.sgd(fevaltrunk, self.pt, self.optimState.trunk)
    optim.sgd(feval, params, optimState)

    -- update loss
    self.lossmeter:add(lossbatch)
  end
  print("Epoch",epoch)
  print("Iterations",iteration,"data loader size",dataloader:size())
  print("Sum loss",moving_loss_average)
  print("Avg loss", moving_loss_average/iteration)

  if config.noisefilter == 'self-paced' then
    self.criterion:increase_threshold()
  end
  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  --torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  print("Epoch ", epoch, epoch%self.config.maxepoch)
  if epoch%self.config.maxepoch == 0 then
    --torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
    --  self.modelsv)
    torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  end

  --self.plotter:info({created_time=io.popen('date'):read(),
  --              tag='My First Plot'})
  --self.plotter:add('Loss', 'training', epoch, self.lossmeter:value())
  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
local maxacc = 0
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  self.maskmeter:reset()
  self.scoremeter:reset()

  for n, sample in dataloader:run() do
    -- copy input and target to the GPU
    self:copySamples(sample)

    if sample.head == 1 then
      local outputs = self.maskNet:forward(self.inputs)
      self.maskmeter:add(outputs:view(self.labels:size()),self.labels)
    else
      local outputs = self.scoreNet:forward(self.inputs)
      self.scoremeter:add(outputs, self.labels)
    end
    cutorch.synchronize()

  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.maskmeter:value('0.7')
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end
  if self.hfreq == 1.0 then
    print("Check who is best")
    z,bestmodel = self.scoremeter:value()
    if z > maxacc then
      print("Best model")
      torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
      maxacc = z
      bestmodel = true
    end
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f suc@.7 %06.2f '..
      '| acc %06.2f | bestmodel %s',
      epoch,
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
      self.scoremeter:value(), bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --self.plotter:add('IoU', 'test mean', epoch, self.maskmeter:value('mean'))
  --self.plotter:add('IoU', 'test median', epoch, self.maskmeter:value('median'))

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
