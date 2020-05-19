--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Train DeepMask or SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('train DeepMask or SharpMask')
cmd:text()
cmd:text('Options:')
cmd:option('-rundir', 'exps/', 'experiments directory')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-nthreads', 2, 'number of threads for DataSampler')
cmd:option('-reload', '', 'reload a network from given directory')
cmd:option('-finetune', '', 'finetune a network from given directory')
cmd:text()
cmd:text('Training Options:')
cmd:option('-batch', 32, 'training batch size')
cmd:option('-lr', 0, 'learning rate (0 uses default lr schedule)')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-wd', 5e-4, 'weight decay')
cmd:option('-maxload', 4000, 'max number of training batches per epoch')
cmd:option('-testmaxload', 500, 'max number of testing batches')
cmd:option('-maxepoch', 300, 'max number of training epochs')
cmd:option('-iSz', 160, 'input size')
cmd:option('-oSz', 56, 'output size')
cmd:option('-gSz', 112, 'ground truth size')
cmd:option('-shift', 16, 'shift jitter allowed')
cmd:option('-scale', .25, 'scale jitter allowed')
cmd:option('-hfreq', 0.5, 'mask/score head sampling frequency')
cmd:option('-scratch', false, 'train DeepMask with randomly initialize weights')
cmd:text()
cmd:text('SharpMask Options:')
cmd:option('-dm', '', 'path to trained deepmask (if dm, then train SharpMask)')
cmd:option('-km', 32, 'km')
cmd:option('-ks', 32, 'ks')
cmd:option('-nobootstrap',false,'remove noisy labels via bootstrapping')
cmd:option('-noisefilter','nofilter','options: nofilter,self-paced')
cmd:option('-coteachforgetrate', 0.2, 'co teaching forget rate tau')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)

local trainSm -- flag to train SharpMask (true) or DeepMask (false)
if #config.dm > 0 then
  trainSm = true
  config.hfreq = 0 -- train only mask head
  config.gSz = config.iSz -- in sharpmask, ground-truth has same dim as input
end

paths.dofile('DeepMask.lua')
if trainSm then paths.dofile('SharpMask.lua') end

--------------------------------------------------------------------------------
-- reload?
local epoch, model
local model2
if #config.reload > 0 then
  epoch = 0
  if paths.filep(config.reload..'/log') then
    for line in io.lines(config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', config.reload))
  local m = torch.load(string.format('%s/model.t7', config.reload))
  model, config = m.model, m.config
end

if #config.finetune > 0 then
  print(string.format('| finetune model %s', config.finetune))
  local m = torch.load(string.format('%s/model.t7', config.finetune))
  model = m.model
  if config.noisefilter == 'coteaching' then
    local m2 = torch.load(string.format('%s/model.t7', config.finetune))
    model2 = m2.model
  end
end

--------------------------------------------------------------------------------
-- directory to save log and model
print(string.format('| running in directory %s', config.rundir))
os.execute(string.format('mkdir -p %s',config.rundir))

--------------------------------------------------------------------------------
-- network and criterion
model = model or (trainSm and nn.SharpMask(config) or nn.DeepMask(config))
if config.noisefilter == 'coteaching' then
  model2 = model2 or (trainSm and nn.SharpMask(config) or nn.DeepMask(config))
end
local criterion
if config.noisefilter == 'self-paced' then
  print("| loss: self-paced learning")
  paths.dofile('SPLCriterion.lua')
  criterion = nn.SPLCriterion(config)
elseif config.noisefilter == 'nofilter' then
  print("| loss: logistic regression without noise filter")
  criterion = nn.SoftMarginCriterion():cuda()
elseif config.noisefilter == 'reedhard' then
  print("| loss: logistic regression with reedhard bootstrapping")
  paths.dofile('ReedHardCriterion.lua')
  criterion = nn.ReedHardCriterion(config)
elseif config.noisefilter == 'mixup' then
  print("| loss: logistic regression with mixup")
  criterion = nn.SoftMarginCriterion():cuda()
elseif config.noisefilter == 'coteaching' then
  print("| loss: logistic regression with co-teaching")
  paths.dofile('CoTeachingCriterion.lua')
  criterion = nn.CoTeachingCriterion()
end
--------------------------------------------------------------------------------
-- initialize data loader
local DataLoader = paths.dofile('DataLoader.lua')
local trainLoader, valLoader = DataLoader.create(config)

--------------------------------------------------------------------------------
-- initialize Trainer (handles training/testing loop)
if trainSm then
  paths.dofile('TrainerSharpMask.lua')
else
  paths.dofile('TrainerDeepMask.lua')
end
local  trainer = Trainer(model,model2, criterion, config)
--------------------------------------------------------------------------------
-- do it
epoch = epoch or 1
print('| start training')
for i = 1, config.maxepoch do
  if config.noisefilter == 'coteaching' then
    trainer:trainCoTeaching(epoch,trainLoader)
    print("Train Co-Teaching epoch:",i)
  else
    trainer:train(epoch,trainLoader)
  end
  if i%1 == 0 then trainer:test(epoch,valLoader) end
  epoch = epoch + 1
end
