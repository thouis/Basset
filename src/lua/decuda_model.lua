#!/usr/bin/env th

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Move a DNA ConvNet off CUDA to the CPU')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('out_file')
cmd:text()
cmd:option('-cudnn', false, 'Model uses cuDNN')
cmd:text()
opt = cmd:parse(arg)

cuda = true
cuda_nn = opt.cudnn
require 'convnet'

----------------------------------------------------------------

-- load parameters
local convnet_params = torch.load(opt.model_file)

-- construct network
local convnet = ConvNet:__init()
convnet:load(convnet_params)

-- move off GPU
convnet:decuda()

-- save back to disk
torch.save(opt.out_file, convnet)
