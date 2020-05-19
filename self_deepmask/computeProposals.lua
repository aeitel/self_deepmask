--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Run full scene inference in sample image
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

local utils = require 'utils'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img','data/testImage.jpg' ,'path/to/test/image')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-np', 100,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-imdir','','path/to/imagedirectory')
cmd:option('-nms', true, 'use non maximum suppression')
cmd:option('-area_max',0.10,'maximum area masks can have with respect to image')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

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
  dm = config.dm,
}

--------------------------------------------------------------------------------
-- do it
print('| start',config.imdir,"Model",config.model)
if config.imdir then
-- load multiple images in directory
  --local resultDir = string.gsub(config.imdir, "/data", "/result")
  local resultDir = string.gsub(config.imdir, "/rgb_frames", string.gsub(config.model, "pretrained", ""))
  print('ResultDir' .. resultDir)
  paths.mkdir(resultDir)
  for f in paths.iterfiles(config.imdir) do
    local empty = false
    local pathImg = string.format('%s/%s',config.imdir,f)
    print('Loading' .. pathImg)
    local img = image.load(pathImg)
    local h,w = img:size(2),img:size(3)
    -- forward all scales
    infer:forward(img)

    -- get top propsals
    local masks,scores = infer:getTopProps(.2,h,w)

    if config.area_max then
      local Rs = maskApi.encode(masks)
      local areas = maskApi.area(Rs)
      local area_threshold = config.area_max*h*w
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
    if config.nms then
      print("Apply non maximum suppression")
      local Rs = maskApi.encode(masks)
      local bboxes = maskApi.toBbox(Rs)
      bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2))
      scores_prob = scores:select(2,1)
      local scored_boxes = torch.cat(bboxes:float(), scores_prob:float(), 2)
      local final_idx = utils.nms_dense(scored_boxes, 0.3)
      -- remove suppressed masks
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
      print("Warning: no crop box found in annotations")
    end
  end
    if not empty then
    -- save result
      local res = img:clone()
      maskApi.drawMasks(res, masks)
      --image.save(string.format('./res.jpg',config.model),res)
      local out = string.gsub(resultDir, "rgb_frames", config.model) .. '/' .. paths.basename(pathImg)
      print('Out' .. out)
      image.save(out,res)
      print('| done')
    end
  end
  collectgarbage()
else
  -- load one image
  local img = image.load(config.img)
  local h,w = img:size(2),img:size(3)

  -- forward all scales
  infer:forward(img)

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
  if config.nms then
    print("Apply non maximum suppression")
    local Rs = maskApi.encode(masks)
    local bboxes = maskApi.toBbox(Rs)
    bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2))
    scores_prob = scores:select(2,1)
    local scored_boxes = torch.cat(bboxes:float(), scores_prob:float(), 2)
    print("Number of masks before NMS", scored_boxes:size(1))
    local final_idx = utils.nms_dense(scored_boxes, 0.3)
    print("NMS final idx",final_idx)
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
      print("Warning: no crop box found in annotations")
    end
  end
  if not empty then
    -- save result
    local res = img:clone()
    maskApi.drawMasks(res, masks, masks:size(1),0.6)
    --image.save(string.format('./res.jpg',config.model),res)
    image.save("./" .. paths.basename(config.img),res)

    print('| done')
  end
  collectgarbage()
end
