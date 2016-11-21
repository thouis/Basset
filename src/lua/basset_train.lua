#!/usr/bin/env th

require 'hdf5'

require 'batcher'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet training')
cmd:text()
cmd:text('Arguments')
cmd:argument('data_file')
cmd:text()
cmd:text('Options:')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-cudnn', false, 'Run on GPGPU w/ cuDNN')
cmd:option('-drop_rate', false, 'Decrease the learning_rate when training loss stalls')
cmd:option('-job', '', 'Table of job hyper-parameters')
cmd:option('-max_epochs', 1000, 'Maximum training epochs to perform')
cmd:option('-restart', '', 'Restart an interrupted training run')
cmd:option('-result', '', 'Write the loss value to this file (useful for Bayes Opt)')
cmd:option('-save', 'dnacnn', 'Prefix for saved models')
cmd:option('-seed', '', 'Seed the model with the parameters of another')
cmd:option('-rand', 1, 'Random number generator seed')
cmd:option('-stagnant_t', 10, 'Allowed epochs with stagnant validation loss')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.rand)

-- set cpu/gpu
cuda_nn = opt.cudnn
cuda = opt.cuda or opt.cudnn
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local data_open = hdf5.open(opt.data_file, 'r')
local train_targets = data_open:read('train_out')
local train_seqs = data_open:read('train_in')
local valid_targets = data_open:read('valid_out')
local valid_seqs = data_open:read('valid_in')

local num_seqs = train_seqs:dataspaceSize()[1]
local init_depth = train_seqs:dataspaceSize()[2]
local seq_len = train_seqs:dataspaceSize()[4]
local num_targets = train_targets:dataspaceSize()[2]

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
local job = {}

-- get parameters
local scientist = nil
if opt.job == '' then
    print("Hyper-parameters unspecified. Applying a small model architecture")
    job.conv_filters = {300,300,500}
    job.conv_filter_sizes = {21,6,4}
    job.pool_width = {4,4,4}

    job.hidden_units = {800}
    job.hidden_dropouts = {0.5}
else
    local job_in = io.open(opt.job, 'r')
    local line = job_in:read()
    while line ~= nil do
        for k, v in string.gmatch(line, "([%w%p]+)%s+([%w%p]+)") do
            -- if key already exsits
            if job[k] then
                -- change to a table
                if type(job[k]) ~= 'table' then
                    job[k] = {job[k]}
                end

                -- write new value to the end
                local jobk_len = #job[k]
                job[k][jobk_len+1] = tonumber(v)
            else
                -- just save the value
                job[k] = tonumber(v)
                if job[k] == nil then
                    job[k] = v
                end
            end
        end
        line = job_in:read()
    end
    job_in:close()

    print(job)
end

-- initialize
local convnet = ConvNet:__init()

local build_success = true
if opt.restart ~= '' then
    local convnet_params = torch.load(opt.restart)
    convnet:load(convnet_params)
    convnet:adjust_optim(job)
elseif opt.seed ~= '' then
    local convnet_params = torch.load(opt.seed)
    convnet:load(convnet_params)
    convnet:adjust_final(num_targets, job.target_type)
    convnet:adjust_optim(job)
else
    build_success = convnet:build(job, init_depth, seq_len, num_targets)

    if build_success == false then
        print('Invalid model')

        -- update spearmint
        if opt.result ~= '' then
            -- print result to file
            local result_out = io.open(opt.result, 'w')
            result_out:write('1000000\n')
            result_out:close()
        end

        os.exit()
    end
end
convnet.model:training()

----------------------------------------------------------------
-- run
----------------------------------------------------------------
local epoch = 1
local epoch_best = 1
local acc_best = 0
local train_loss_last
local valid_loss
local valid_acc
local batcher = Batcher:__init(train_seqs, train_targets, convnet.batch_size)

print(tostring(convnet.model.criterion))
print("bar")

while epoch <= opt.max_epochs and epoch - epoch_best <= opt.stagnant_t do
    io.write(string.format("Epoch #%3d   ", epoch))
    local start_time = sys.clock()

    -- conduct one training epoch
    local train_loss = convnet:train_epoch(batcher)
    io.write(string.format("train loss = %7.3f, ", train_loss))

    if job.mc_n ~= nil and job.mc_n > 1 then
        -- measure accuracy on a test set
        valid_loss, valid_acc, valid_cor = convnet:test_mc(valid_seqs, valid_targets, job.mc_n)

    else
        -- change to evaluate mode
        convnet.model:evaluate()

        -- measure accuracy on a test set
        valid_loss, valid_acc, valid_cor = convnet:test(valid_seqs, valid_targets)
    end

    local valid_acc_avg = torch.mean(valid_acc)
    local acc_str
    if convnet.target_type == "binary" then
        acc_str = string.format("AUC = %.4f", valid_acc_avg)
    else
        local valid_cor_avg = torch.mean(valid_cor)
        acc_str = string.format("R2 = %.4f, rho = %.4f", valid_acc_avg, valid_cor_avg)
    end

    -- print w/ time
    local epoch_time = sys.clock()-start_time
    if epoch_time < 600 then
        time_str = string.format('%3ds', epoch_time)
    else
        time_str = string.format('%3dm', epoch_time/60)
    end

    io.write(string.format("valid loss = %7.3f, %s, time = %s", valid_loss, acc_str, time_str))

    -- save checkpoint
    convnet:sanitize()
    torch.save(string.format('%s_check.th' % opt.save), convnet)

    -- update best
    if valid_acc_avg > acc_best then
        io.write(" best!")
        acc_best = valid_acc_avg
        epoch_best = epoch

        -- save best
        torch.save(string.format('%s_best.th' % opt.save), convnet)
    end

    -- drop learning rate
    if opt.drop_rate and train_loss_last ~= nil and (train_loss_last - train_loss)/train_loss_last < .001 then
        convnet:drop_rate()
        io.write(", rate drop")
    end
    train_loss_last = train_loss

    -- change back to training mode
    convnet.model:training()

    -- increment epoch
    epoch = epoch + 1

    print('')
end

if opt.result ~= '' then
    -- print result to file
    local result_out = io.open(opt.result, 'w')
    result_out:write(acc_best, '\n')
    result_out:close()
end

data_open:close()
