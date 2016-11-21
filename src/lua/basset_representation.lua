#!/usr/bin/env th

require 'hdf5'
require 'lfs'

require 'batcher'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Extract the final representation in a ConvNet for the set of sequences')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-batch', 128, 'Batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-cudnn', false, 'Run on GPGPU w/ cuDNN')
cmd:text()
opt = cmd:parse(arg)

-- set cpu/gpu
cuda_nn = opt.cudnn
cuda = opt.cuda or opt.cudnn
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local data_open = hdf5.open(opt.data_file, 'r')
local train_seqs = data_open:read('train_in')
local valid_seqs = data_open:read('valid_in')
local test_seqs = data_open:read('test_in')

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
-- initialize
local convnet = ConvNet:__init()

-- load from saved parameters
local convnet_params = torch.load(opt.model_file)
convnet:load(convnet_params)

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- guarantee evaluate mode
convnet.model:evaluate()

-- get representations
local train_repr = convnet:predict_finalrepr(train_seqs, opt.batch_size)
local valid_repr = convnet:predict_finalrepr(valid_seqs, opt.batch_size)
local test_repr = convnet:predict_finalrepr(test_seqs, opt.batch_size)

data_open:close()

----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('train_repr', train_repr)
hdf_out:write('valid_repr', valid_repr)
hdf_out:write('test_repr', test_repr)
hdf_out:close()
