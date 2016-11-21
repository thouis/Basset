#!/usr/bin/env th

require 'hdf5'

require 'convnet_io'
require 'postprocess'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet response to DB motifs')
cmd:text()
cmd:text('Arguments')
cmd:argument('motifs_file')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-add_motif', '', 'Additional motif to pre-insert')
cmd:option('-batch_size', 256, 'Maximum batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(1)

cuda = opt.cuda
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)
convnet.model:evaluate()

-- open HDF5 and get test sequences
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in'):all()

local num_seqs = (#test_seqs)[1]
local seq_len = (#test_seqs)[4]
local seq_mid = seq_len/2 - 5

local motifs_open = hdf5.open(opt.motifs_file, 'r')
local motifs = motifs_open:all()
local num_motifs = 0
for mid, _ in pairs(motifs) do
    num_motifs = num_motifs + 1
end

----------------------------------------------------------------
-- write in the additional motif
----------------------------------------------------------------
if #opt.add_motif > 0 then
    seq_add = seq_mid - 2*(#opt.add_motif)
    for si = 1,num_seqs do
        for pi = 1,#opt.add_motif do
            -- determine nt index
            if opt.add_motif:sub(pi,pi) == 'A' then
                nt = 1
            elseif opt.add_motif:sub(pi,pi) == 'C' then
                nt = 2
            elseif opt.add_motif:sub(pi,pi) == 'G' then
                nt = 3
            else
                nt = 4
            end

             -- set the nt
            for ni = 1,4 do
                test_seqs[si][ni][1][seq_add+pi] = 0
            end
            test_seqs[si][nt][1][seq_add+pi] = 1
        end
    end
end

----------------------------------------------------------------
-- predict
----------------------------------------------------------------
-- make initial predictions
local preds, scores, reprs = convnet:predict_reprs(test_seqs, opt.batch_size, true)

-- normalize
troy_norm(preds, convnet.pred_means)

-- initialize difference storage
local num_targets = (#preds)[2]
local scores_diffs = torch.Tensor(num_motifs, num_targets)
local preds_diffs = torch.Tensor(num_motifs, num_targets)
local reprs_diffs = {}
for l = 1,#reprs do
    reprs_diffs[l] = torch.Tensor(num_motifs, (#reprs[l])[2])
end

-- compute score mean and variance
local scores_means = scores:mean(1):squeeze()
-- local scores_stds = scores:std(1):squeeze()
local preds_means = preds:mean(1):squeeze()

-- compute hidden unit means
local reprs_means = {}
for l = 1,#reprs do
    if reprs[l]:nDimension() == 2 then
        -- fully connected
        reprs_means[l] = reprs[l]:mean(1):squeeze()
    else
        -- convolution
        reprs_means[l] = reprs[l]:mean(3):mean(1):squeeze()
    end
end

-- for each motif
for mi = 1,num_motifs do
    print(mi)

    -- copy the test seqs
    local test_seqs_motif = test_seqs:clone()

    -- access motif
    local motif = motifs[tostring(mi)]

    for si = 1,num_seqs do
        -- sample a motif sequence
        for pi = 1,(#motif)[2] do
            -- choose a random nt
            local r = torch.uniform() - 0.0001
            local nt = 1
            local psum = motif[nt][pi]
            while psum < r do
                nt = nt + 1
                psum = psum + motif[nt][pi]
            end

            -- set the nt
            for ni = 1,4 do
                test_seqs_motif[si][ni][1][seq_mid+pi] = 0
            end
            test_seqs_motif[si][nt][1][seq_mid+pi] = 1
        end
    end

    -- predict
    local mpreds, mscores, mreprs = convnet:predict_reprs(test_seqs_motif, opt.batch_size, true)

    -- normalize
    troy_norm(preds, convnet.pred_means)

    -- compute stats
    local mscores_means = mscores:mean(1):squeeze()
    local mpreds_means = mpreds:mean(1):squeeze()
    local mreprs_means = {}
    for l = 1,#reprs do
        if mreprs[l]:nDimension() == 2 then
            -- fully connected
            mreprs_means[l] = mreprs[l]:mean(1):squeeze()
        else
            -- convolution
            mreprs_means[l] = mreprs[l]:mean(3):mean(1):squeeze()
        end
    end

    -- save difference
    scores_diffs[mi] = mscores_means - scores_means
    preds_diffs[mi] = mpreds_means - preds_means

    -- compute a statistical test?

    -- repr difference
    for l = 1,#reprs do
        reprs_diffs[l][mi] = mreprs_means[l] - reprs_means[l]
    end
end

----------------------------------------------------------------
-- dump to file, load into python
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('scores_diffs', scores_diffs)
hdf_out:write('preds_diffs', preds_diffs)
for l = 1,#reprs_diffs do
    local repr_name = string.format("reprs%d", l)
    hdf_out:write(repr_name, reprs_diffs[l])
end
hdf_out:close()
