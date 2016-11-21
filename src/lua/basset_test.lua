#!/usr/bin/env th

require 'hdf5'
require 'lfs'

require 'batcher'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet testing')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_dir')
cmd:text()
cmd:text('Options:')
cmd:option('-batch', 128, 'Batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-cudnn', false, 'Run on GPGPU w/ cuDNN')
cmd:option('-rc', false, 'Average forward and reverse complement')
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
local test_targets = data_open:read('test_out')
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

-- measure accuracy on a test set
local loss, AUCs, roc_points = convnet:test(test_seqs, test_targets, opt.batch_size, opt.rc)
local avg_auc = torch.mean(AUCs)

-- cd to output dir
local rs = lfs.chdir(opt.out_dir) and true or false
if not rs then
    lfs.mkdir(opt.out_dir)
    lfs.chdir(opt.out_dir)
end

-- print AUCs
local auc_out = io.open('aucs.txt', 'w')
for i=1,(#AUCs)[1] do
    auc_out:write(string.format("%-3d   %.4f\n", i, AUCs[i]))
end
auc_out:close()

print(string.format("mean  %.4f", avg_auc))
print(string.format("loss  %.3f", loss))

-- print ROC points
for yi=1,#roc_points do
    local roc_file = string.format('roc%d.txt', yi)
    local roc_out = io.open(roc_file, 'w')
    if roc_points[yi] ~= nil then
        for pi = 1,(#(roc_points[yi]))[1] do
            roc_out:write(string.format("%f\t%f\n", roc_points[yi][pi][1], roc_points[yi][pi][2]))
        end
    end
    roc_out:close()
end

data_open:close()