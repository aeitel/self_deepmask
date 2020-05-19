--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Full scene evaluation of DeepMask/SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

local cjson = require 'cjson'
local tds = require 'tds'
local coco = require 'coco'
local utils = require 'utils'

paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('full scene evaluation of DeepMask/SharpMask')
cmd:text()
cmd:argument('-model', 'model to load')
cmd:text('Options:')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-split', 'val', 'dataset split to be used (train/val)')
cmd:option('-np', 100,'number of proposals')
cmd:option('-thr', .2, 'mask binary threshold')
cmd:option('-save', true, 'save top proposals')
cmd:option('-startAt', 1, 'start image id')
--cmd:option('-endAt', 50, 'end image id') -- use dataset length
cmd:option('-smin', -2.5, 'min scale')
cmd:option('-smax', .5, 'max scale')
cmd:option('-sstep', .5, 'scale step')
cmd:option('-timer', false, 'breakdown timer')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-nms_overlap', 0.3, 'use non maximum suppression')
cmd:option('-area_max',0.1,'maximum area masks can have with respect to image')
cmd:option ('-crop_workspace',false,'remove segments outside of workspace (category_id=2)')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)
local maskApi = coco.MaskApi
local meanstd = {mean={ 0.485, 0.456, 0.406 }, std={ 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load model and config
print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/bestmodel.t7')
--local m = torch.load(config.model..'/model.t7')
--print('Loading model from last epoch')
local c = m.config
for k,v in pairs(c) do if config[k] == nil then config[k] = v end end
local epoch = 0
if paths.filep(config.model..'/log') then
  for line in io.lines(config.model..'/log') do
    if string.find(line,'train') then epoch = epoch + 1 end
  end
  print(string.format('| number of examples seen until now: %d (%d epochs)',
    epoch*config.maxload*config.batch,epoch))
end

local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- directory to save results
--local savedir = string.format('%s/epoch=%d/',config.model,epoch)
local savedir = string.format('%s/%s/%s',config.model, config.datadir:match( "([^/]+)$" ), config.split)
if config.nms_overlap > 0.0 then
  --savedir = string.format('%s_%s/epoch=%d/',config.model,'nms',epoch)
  savedir = string.format('%s_%s%s/%s/%s',config.model,'nms',config.nms_overlap, config.datadir:match( "([^/]+)$" ), config.split)
  print(savedir)
end

print(string.format('| saving results results in %s',savedir))
os.execute(string.format('mkdir -p %s',savedir))
os.execute(string.format('mkdir -p %s/t7',savedir))
os.execute(string.format('mkdir -p %s/jsons',savedir))
if config.save then os.execute(string.format('mkdir -p %s/res',savedir)) end
if config.save then os.execute(string.format('mkdir -p %s/res_%s',savedir,config.split)) end

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.smin,config.smax,config.sstep do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  iSz = config.iSz,
  dm = config.dm,
  timer = config.timer,
}

--------------------------------------------------------------------------------
-- get list of eval images
local annFile = string.format('%s/annotations/instances_%02s.json',
  config.datadir,config.split)
print(annFile)
local coco = coco.CocoApi(annFile)
local imgIds = coco:getImgIds()
imgIds,_ = imgIds:sort()

--------------------------------------------------------------------------------
-- function: encode proposals
local function encodeProps(props,np,img,k,masks,scores, t)
  --local t = (k-1)*np
  local enc = maskApi.encode(masks)
  for i = 1, np do
    local elem = tds.Hash()
    elem.segmentation = tds.Hash(enc[i])
    elem.image_id=img.id
    elem.category_id=1
    elem.score=scores[i][1]
    props[t+i] = elem
  end
end

--------------------------------------------------------------------------------
-- function: convert props to json and save
local function saveProps(props,savedir,s,e)
  --t7
  local pathsvt7 = string.format('%s/t7/props-%d-%d.t7', savedir,s,e)
  torch.save(pathsvt7,props)
  --json
  local pathsvjson = string.format('%s/jsons/props-%d-%d.json', savedir,s,e)
  local propsjson = {}
  for _,prop in pairs(props) do -- hash2table
    local elem = {}
    elem.category_id = prop.category_id
    elem.image_id = prop.image_id
    elem.score = prop.score
    elem.segmentation={
      size={prop.segmentation.size[1],prop.segmentation.size[2]},
      counts = prop.segmentation.counts or prop.segmentation.count
    }
    table.insert(propsjson,elem)
  end
  local jsonText = cjson.encode(propsjson)
  local f = io.open(pathsvjson,'w'); f:write(jsonText); f:close()
  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: read image
local function readImg(datadir,split,fileName)
  --local pathImg = string.format('%s/%s/%s',datadir,split,fileName)
  local pathImg = string.format('%s/%s',datadir,fileName)

  local inp = image.load(pathImg,3)
  return inp
end

--------------------------------------------------------------------------------
-- run
print('| start eval')
local props, svcount = tds.Hash(), config.startAt
config.endAt= imgIds:size(1)
print(config.endAt)
local np_counter = 0
for k = config.startAt,config.endAt do
  xlua.progress(k,config.endAt)

  -- load image
  local img = coco:loadImgs(imgIds[k])[1]
  local input = readImg(config.datadir,config.split,img.file_name)
  local h,w = img.height,img.width

  -- forward all scales
  infer:forward(input)

  -- get top proposals
  local masks,scores = infer:getTopProps(config.thr,h,w)
  local empty = false

  if config.area_max then
    local Rs = maskApi.encode(masks)
    local areas = maskApi.area(Rs)
    local area_threshold = config.area_max*img.height*img.width
    final_idx = utils.filter_large_masks(areas, area_threshold)
    if final_idx:nDimension()>0 then
      masks = masks:index(1,final_idx)
      scores = scores:index(1,final_idx)
      areas = areas:index(1,final_idx)
      _, indices = torch.sort(scores:select(2,1),1,true)
      areas = areas:index(1,indices)
      scores = scores:index(1,indices)
      masks = masks:index(1,indices)
    else
      emtpy = true
    end
  end

  -- NMS
  if config.nms_overlap > 0.0 then
    print("Apply non maximum suppression",config.nms_overlap)
    local Rs = maskApi.encode(masks)
    local bboxes = maskApi.toBbox(Rs)
    bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2))
    scores_prob = scores:select(2,1)
    local scored_boxes = torch.cat(bboxes:float(), scores_prob:float(), 2)
    print("Number of masks before NMS", scored_boxes:size(1))
    local final_idx = utils.nms_dense(scored_boxes, config.nms_overlap)
    --print("NMS final idx",final_idx)
    if final_idx:nDimension()>0 then
      -- remove suppressed masks
      masks = masks:index(1, final_idx)
      scores = scores:index(1, final_idx)
      bboxes = bboxes:index(1, final_idx)
    else
      empty = true
    end
  end

  if config.crop_workspace then
    print("Filter out masks outside of workspace")
    local annIds = coco:getAnnIds({imgId=imgIds[k]})
    local anns = coco:loadAnns(annIds)

    local R, crop_box
    for i=1,#anns do
      -- assumes one workspace instance with category_id=2
      if anns[i].category_id ==2 then
        R = maskApi.frBbox(anns[i].bbox,h,w)[1]
        crop_box = anns[i].bbox
      end
    end
    if crop_box then
      -- Visualize workspace box
      --local gt_mask = maskApi.decode(R)
      --local O = input:clone():contiguous():float()
      --maskApi.drawMasks(O,maskApi.decode(R),nil,alpha,clrs)
      --image.save('CHECK_'.. k .. '.png',O:double())

      local Rs = maskApi.encode(masks)
      local bboxes = maskApi.toBbox(Rs)
      local final_idx = utils.crop_boxes(bboxes, crop_box, w, h)
      print("Final idx crop",final_idx:nDimension())
      if final_idx:nDimension()>0 then
      -- remove cropped out masks
        masks = masks:index(1, final_idx)
        scores = scores:index(1, final_idx)
        bboxes = bboxes:index(1, final_idx)
      else
        empty=true
      end
    else
      print("No crop box found in annotations")
    end
  end

  if not empty then
    -- encode proposals
    local np = math.min(masks:size(1), config.np)
    encodeProps(props,np,img,k,masks,scores, np_counter)

    -- save top masks?
    if config.save then
      local res = input:clone()
      local objectness_treshold = 0.5
      if objectness_treshold > 0.0 then
        sorted_scores, final_idx = torch.sort(scores:select(2,1), true)
        final_idx = final_idx[torch.ge(sorted_scores,objectness_treshold)]
        maskApi.drawMasks(res, masks:index(1, final_idx), final_idx:size(1), 0.6)
      else
        maskApi.drawMasks(res, masks, masks:size(1), 0.6)
      end
      image.save(string.format('%s/res_%s/%02d.jpg',savedir,config.split,k),res)
    end
    np_counter = np_counter + np

  end

  -- save proposals
  if k%config.endAt == 0 then
    saveProps(props,savedir,svcount,k); props = tds.Hash(); collectgarbage()
    svcount = svcount + config.endAt
    print("Number of saved masks: ",np_counter)
  end

  collectgarbage()
end

if config.timer then infer:printTiming() end
collectgarbage()
print('| finish')
