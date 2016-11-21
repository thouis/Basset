#!/usr/bin/env th

require 'dp'
require 'hdf5'

require 'batcher'
require 'convnet_io'
require 'convnet'
require 'postprocess'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet hidden layer visualizations')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-batch_size', 1000, 'Batch size of sequences per compute')
cmd:option('-seqs', false, 'Print affect on all sequences')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(2)

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)
convnet.model:evaluate()

local num_filters = convnet.conv_filters[1]
local final_i = #convnet.model.modules - 1
if convnet.target_type == "continuous" then
    final_i = final_i + 1
end
local conv_i = 1
while tostring(convnet.model.modules[conv_i]) ~= "nn.ReLU" do
    conv_i = conv_i + 1
end

-- open HDF5 and get test sequences
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in')
local test_targets = data_open:read('test_out')

local num_seqs = test_seqs:dataspaceSize()[1]
local num_targets = test_targets:dataspaceSize()[2]

----------------------------------------------------------------
-- allocate data structures
----------------------------------------------------------------
local filter_means = torch.Tensor(num_filters):zero()
local filter_stds = torch.Tensor(num_filters):zero()
local filter_infl = torch.Tensor(num_filters):zero()
local filter_infl_targets = torch.Tensor(num_filters, num_targets):zero()
local seq_filter_targets
if opt.seqs then
    seq_filter_targets = torch.Tensor(num_seqs, num_filters, num_targets):zero()
end

----------------------------------------------------------------
-- predict
----------------------------------------------------------------
-- setup batcher
local batcher = Batcher:__init(test_seqs, test_targets, opt.batch_size)
local Xb, Yb = batcher:next()
local batches = 1
local sfti = 0

-- while batches remain
while Xb ~= nil do
    --------------------------------------------------
    -- predict and collect statistics
    --------------------------------------------------
    local preds = convnet:predict(Xb, opt.batch_size, true)
    local loss = convnet.criterion(preds:double(), Yb)

	-- compute pred means for target deltas
	local preds_means = preds:mean(1)

    -- normalize with Troy method
    local preds_tnorm = troy_norm(preds, preds_means)

    -- compute final layer stats
    local final_output = convnet.model.modules[final_i].output:clone()
    local final_means = final_output:mean(1):reshape(num_targets)
    local final_stds = final_output:std(1):reshape(num_targets)
    local final_var = torch.pow(final_stds,2)

    --------------------------------------------------
    -- collect filter statistics
    --------------------------------------------------
     -- grab convolution module
    local conv_module = convnet.model.modules[conv_i]

    -- save original
    local actual_rep = conv_module.output:squeeze():clone()

    local batch_size = (#actual_rep)[1]
    local seq_len = (#actual_rep)[3]

    -- compute filter means
    local filter_means_batch = actual_rep:mean(1):mean(3):squeeze()
    filter_means = filter_means + filter_means_batch

    -- compute filter stds
    local filters_out1 = torch.swapaxes(actual_rep,{2,1,3}):reshape(num_filters,batch_size*seq_len)
    filter_stds = filter_stds + filters_out1:std(2):squeeze()

    --------------------------------------------------
    -- nullify and re-measure
    --------------------------------------------------
	for fi=1,num_filters do
		print(string.format(" Filter %d", fi))

		-- if the unit is inactive
		local zloss = loss
        local zpreds = preds
        local zfinal_output = final_output
        local zfinal_means = final_means

		if filter_means_batch[fi] == 0 then
			print("  Inactive")
		else
			-- zero the hidden unit
			local zero_rep = actual_rep:clone()
			zero_rep[{{},fi,{}}] = filter_means_batch[fi]
			conv_module.output = zero_rep:reshape(batch_size, num_filters, 1, seq_len)

			-- propagate the change through the network
			local currentOutput = conv_module.output
			for mz=conv_i+1,#convnet.model.modules do
				currentOutput = convnet.model.modules[mz]:updateOutput(currentOutput)
			end
			zpreds = currentOutput
            zfinal_output = convnet.model.modules[final_i].output
            zfinal_means = zfinal_output:mean(1):reshape(num_targets)

			-- re-compute loss
			zloss = convnet.criterion:forward(zpreds:double(), Yb)

			-- reset hidden unit
			conv_module.output = actual_rep:reshape(batch_size, num_filters, 1, seq_len)
		end

		-- save loss differences

        -- prediction troy-normalized difference
        -- local zpreds_tnorm = troy_norm(zpreds, preds_means)
        -- layer_target_deltas[li][i] = layer_target_deltas[li][i] + (preds_tnorm - zpreds_tnorm):mean(1):squeeze()

        -- final z normalized difference
        local batch_infl_targets = (final_means - zfinal_means)
        batch_infl_targets:cdiv(final_var)
        filter_infl_targets[fi] = filter_infl_targets[fi] + batch_infl_targets

		-- save unit delta
		filter_infl[fi] = filter_infl[fi] + (zloss - loss)
        print(filter_infl[fi])

        -- save sequence effects
        if opt.seqs then
            for bi=1,batch_size do
                seq_filter_targets[sfti+bi][fi] = final_output[bi] - zfinal_output[bi]
            end
        end

		collectgarbage()
	end

    -- advance sequence index
    if opt.seqs then
        sfti = sfti + batch_size
    end

	-- next batch
	Xb, Yb = batcher:next()
    batches = batches + 1

end

-- close HDF5
data_open:close()

-- go back one
batches = batches - 1

-- normalize by batch number
filter_means = filter_means / batches
filter_stds = filter_stds / batches
filter_infl_targets = filter_infl_targets / batches

-- normalize by sequences
filter_infl = filter_infl / num_seqs

-- dump to file, load into python
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('filter_means', filter_means)
hdf_out:write('filter_stds', filter_stds)
hdf_out:write('filter_infl', filter_infl)
hdf_out:write('filter_infl_targets', filter_infl_targets)
if opt.seqs then
    hdf_out:write('seq_filter_targets', seq_filter_targets)
end
hdf_out:close()
