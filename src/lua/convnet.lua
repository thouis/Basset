require 'dpnn'
require 'nn'
require 'optim'

if cuda then
    require 'cunn'
    require 'cutorch'
    -- require 'inn'
end
if cuda_nn then
    require 'cudnn'
end

require 'accuracy'
require 'batcher'
require 'batcherT'


ConvNet = {}

function ConvNet:__init()
    obj = {}
    setmetatable(obj, self)
    self.__index = self
    return obj
end

----------------------------------------------------------------
-- adjust_final
--
-- Reset the final layer for new targets.
----------------------------------------------------------------
function ConvNet:adjust_final(num_targets, target_type)
    -- re-specify num targets
    self.num_targets = num_targets

    -- re-specify target_type
    local target_type_seed = self.target_type
    self.target_type = target_type or self.target_type

    -- remove nonlinearity
    if target_type_seed == "binary" or target_type_seed == "positive" then
        self.model:remove()
    end

     -- save input size
    local hidden_in = self.model.modules[#self.model].weight:size(2)

    -- remove Linear
    self.model:remove()

    -- add final linear
    self.model:add(nn.Linear(hidden_in, self.num_targets))

    if self.target_type == "binary" then
        -- add final nonlinearity
        self.model:add(nn.Sigmoid())

        -- binary cross-entropy loss
        self.criterion = nn.BCECriterion()

    elseif self.target_type == "positive" then
        -- binary cross-entropy loss
        self.model:add(nn.ReLU())

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()

    else
        -- mean-squared error loss
        self.criterion = nn.MSECriterion()
    end

    self.criterion.sizeAverage = false

    -- cuda
    if cuda then
        print("Running on GPU.")
        self.model:cuda()
        self.criterion:cuda()
    end
    if cuda_nn then
        self.model = cudnn.convert(self.model, cudnn)
    end

    -- retrieve parameters and gradients
    self.parameters, self.gradParameters = self.model:getParameters()

    -- print model summary
    print(self.model)
end

----------------------------------------------------------------
-- adjust_optim
--
-- Reset the optim state and set optimization hyper-params
----------------------------------------------------------------
function ConvNet:adjust_optim(job)
    -- number of examples per weight update
    self.batch_size = job.batch_size or 128

    -- base learning rate
    self.learning_rate = job.learning_rate or 0.002
    self.optim_state.learningRate = self.learning_rate

    if self.optimization == "rmsprop" then
        -- reset state
        self.optim_state.m = nil

        -- gradient momentum
        self.momentum = job.momentum or 0.98
        self.optim_state.alpha = self.momentum

    elseif self.optimization == "adam" then
        -- reset state
        self.optim_state.t = nil
        self.optim_state.m = nil
        self.optim_state.v = nil
        self.optim_state.denom = nil

        -- gradients momentum
        self.beta1 = job.beta1 or 0.9
        self.optim_state.beta1 = self.beta1
        self.beta2 = job.beta2 or 0.999
        self.optim_state.beta2 = self.beta2

    else
        print("Unrecognized optimization algorithm")
        exit(1)
    end
end


----------------------------------------------------------------
-- build
--
-- Build the network using the job parameters and data
-- attributes.
----------------------------------------------------------------
function ConvNet:build(job, init_depth, init_len, num_targets)
    -- parse network structure parameters
    self:setStructureParams(job)

    -- initialize model sequential
    self.model = nn.Sequential()

    -- store useful values
    self.num_targets = num_targets
    local depth = init_depth
    local seq_len = init_len

    -- convolution layers
    for i = 1,self.conv_layers do
        -- convolution
        if i == 1 or self.conv_conn[i-1] == 1 then
            -- TEMP padding
            out_width = math.ceil(seq_len / self.conv_filter_strides[i])
            pad_width = (out_width-1) * self.conv_filter_strides[i] + self.conv_filter_sizes[i] - seq_len
            pad_left = math.floor(pad_width / 2)
            print(string.format("seq_len: %d, filter_size: %d, pad_width: %d", seq_len, self.conv_filter_sizes[i], pad_width))

            -- fully connected convolution
            -- self.model:add(nn.SpatialConvolution(depth, self.conv_filters[i], self.conv_filter_sizes[i], 1))
            self.model:add(nn.SpatialConvolution(depth, self.conv_filters[i], self.conv_filter_sizes[i], 1, self.conv_filter_strides[i], 1, pad_left, 0))

        else
            -- randomly connected convolution
            num_to = torch.round(depth*self.conv_conn[i-1])
            conn_matrix = nn.tables.random(depth, self.conv_filters[i], num_to)
            self.model:add(nn.SpatialConvolutionMap(conn_matrix, self.conv_filter_sizes[i], 1))
        end

        -- TEMP: padding instead
        -- update sequence length for filter pass
        -- seq_len = seq_len - self.conv_filter_sizes[i] + 1

        -- batch normalization (need to figure out how to ditch the bias above)
        if self.batch_normalize then
            self.model:add(nn.SpatialBatchNormalization(self.conv_filters[i]))
        end

        -- nonlinearity
        self.model:add(nn.ReLU())

        -- dropout
        if self.conv_dropouts[i] > 0 then
            self.model:add(nn.Dropout(self.conv_dropouts[i]))
        end

        -- pooling
        if self.pool_width[i] > 1 then
            if self.pool_op == "max" then
                -- trimming the seq
                pseq_len = math.ceil(seq_len / self.pool_width[i])
                pool_mod = nn.SpatialMaxPooling(self.pool_width[i], 1)
                pool_mod:ceil()
                self.model:add(pool_mod)
            else
                pseq_len = math.floor(seq_len / self.pool_width[i])
                self.model:add(inn.SpatialStochasticPooling(self.pool_width[i],1))
            end

            seq_len = pseq_len
        end

        -- update helper
        depth = self.conv_filters[i]
    end

    -- too much pooling
    if seq_len <= 0 then
        return false
    end

    -- prep for fully connected layers
    hidden_in = depth*seq_len
    self.model:add(nn.Reshape(hidden_in))

    -- fully connected hidden layers
    for i =1,self.hidden_layers do
        -- linear transform
        self.model:add(nn.Linear(hidden_in, self.hidden_units[i]))

        -- batch normalization (need to figure out how to ditch the bias above)
        if self.batch_normalize then
            self.model:add(nn.BatchNormalization(self.hidden_units[i]))
        end

        -- nonlinearity
        self.model:add(nn.ReLU())

        -- dropout
        if self.hidden_dropouts[i] > 0 then
            self.model:add(nn.Dropout(self.hidden_dropouts[i]))
        end

        -- update helper
        hidden_in = self.hidden_units[i]
    end

    if self.target_type == "binary" then
        -- final layer w/ target priors as initial biases
        final_linear = nn.Linear(hidden_in, self.num_targets)
        -- target_priors = targets:mean(1):squeeze()
        -- biases_init = -torch.log(torch.pow(target_priors, -1) - 1)
        -- final_linear.bias = biases_init
        self.model:add(final_linear)
        self.model:add(nn.Sigmoid())

        -- binary cross-entropy loss
        self.criterion = nn.BCECriterion()

    elseif self.target_type == "positive" then
        -- final layer
        self.model:add(nn.Linear(hidden_in, self.num_targets))
        self.model:add(nn.ReLU())

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()

    else
        -- final layer
        self.model:add(nn.Linear(hidden_in, self.num_targets))

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()
    end

    self.criterion.sizeAverage = false

    -- cuda
    if cuda then
        print("Running on GPU.")
        self.model:cuda()
        self.criterion:cuda()
    end
    if cuda_nn then
        self.model = cudnn.convert(self.model, cudnn)
    end

    -- retrieve parameters and gradients
    self.parameters, self.gradParameters = self.model:getParameters()

    -- print model summary
    print(self.model)

    -- the following code breaks the program, but
    -- it's interesting to see those counts!

    -- print(string.format("Sum:      %7d parameters",(#self.parameters)[1]))
    -- for i = 1,(#self.model) do
    -- 	local layer_params = self.model.modules[i]:getParameters()
    -- 	local np = 0
    -- 	if layer_params:nDimension() > 0 then
    -- 		np = (#layer_params)[1]
    -- 	end
    -- 	print(string.format("Layer %2d: %7d", i, np))
    -- end

    return true
end


----------------------------------------------------------------
-- cuda
--
-- Move the model to the GPU. Untested.
----------------------------------------------------------------
function ConvNet:cuda()
    if self.optimization == "rmsprop" then
        self.optim_state.m = self.optim_state.m:cuda()
        self.optim_state.tmp = self.optim_state.tmp:cuda()
    elseif self.optimization == "adam" then
        self.optim_state.m = self.optim_state.m:cuda()
        self.optim_state.v = self.optim_state.v:cuda()
        self.optim_state.denom = self.optim_state.denom:cuda()
    end
    self.criterion:cuda()
    self.model:cuda()

    if cuda_nn then
        cudnn.convert(self.model, nn)
    end

    self.parameters, self.gradParameters = self.model:getParameters()
    -- self.parameters = self.parameters:double()
    -- self.gradParameters = self.gradParameters:double()

    cuda = true
end

----------------------------------------------------------------
-- decuda
--
-- Move the model back to the CPU.
----------------------------------------------------------------
function ConvNet:decuda()
    if self.optimization == "rmsprop" then
        self.optim_state.m = self.optim_state.m:double()
        self.optim_state.tmp = self.optim_state.tmp:double()
    elseif self.optimization == "adam" then
        self.optim_state.m = self.optim_state.m:double()
        self.optim_state.v = self.optim_state.v:double()
        self.optim_state.denom = self.optim_state.denom:double()
    end

    self.criterion:double()

    if cuda_nn then
        cudnn.convert(self.model, nn)
    end
    self.model:double()

    self.parameters, self.gradParameters = self.model:getParameters()
    -- self.parameters = self.parameters:double()
    -- self.gradParameters = self.gradParameters:double()

    cuda = false
end


----------------------------------------------------------------
-- drop_rate
--
-- Decrease the optimization learning_rate by a multiplier
----------------------------------------------------------------
function ConvNet:drop_rate()
    self.learning_rate = self.learning_rate * 2/3
    self.optim_state.learningRate = self.learning_rate
end


----------------------------------------------------------------
-- get_nonlinearity
--
-- Return the module representing nonlinearity x.
----------------------------------------------------------------
function ConvNet:get_nonlinearity(x)
    local nl_modules = self.model:findModules('nn.ReLU')
    return nl_modules[x]
end

function ConvNet:get_nonlinears(pool)
    local relu_name = 'nn.ReLU'
    if cuda_nn then
        relu_name = 'cudnn.ReLU'
    end

    local nl_modules = {}
    local ni = 1
    local mi = 1
    local conv_name
    local conv_layers = self.conv_layers
    if pool then
        conv_name = 'nn.SpatialMaxPooling'
        if self.pool_width[1] == 1 then
            conv_layers = self.conv_layers - 1
        end
    else
        conv_name = relu_name
    end

    -- add convolutions
    for c = 1,conv_layers do
        while mi <= #self.model.modules and torch.typename(self.model.modules[mi]) ~= conv_name do
            mi = mi + 1
        end
        if torch.typename(self.model.modules[mi]) ~= conv_name then
            error("Cannot find convolution modules " .. conv_name)
        end
        nl_modules[ni] = self.model.modules[mi]
        ni = ni + 1
        mi = mi + 1
    end

    -- add fully connected
    while mi <= #(self.model.modules) do
        if torch.typename(self.model.modules[mi]) == relu_name then
            nl_modules[ni] = self.model.modules[mi]
            ni = ni + 1
        end
        mi = mi + 1
    end

    return nl_modules
end

----------------------------------------------------------------
-- get_final
--
-- Return the module representing the final layer.
----------------------------------------------------------------
function ConvNet:get_final()
    local layers = #self.model
    return self.model.modules[layers-1]
end


function ConvNet:load(cnn)
    for k, v in pairs(cnn) do
        self[k] = v
    end
end


----------------------------------------------------------------
-- predict
--
-- Predict targets for a new set of sequences.
----------------------------------------------------------------
function ConvNet:predict(Xf, batch_size, Xtens, rc_avg)
    local bs = batch_size or self.batch_size
    local batcher
    if Xtens then
        batcher = BatcherT:__init(Xf, nil, bs)
    else
        batcher = Batcher:__init(Xf, nil, bs)
    end

    -- find final model layer
    local final_i = #self.model.modules - 1
    if self.target_type == "continuous" then
        final_i = final_i + 1
    end

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local scores = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- collect garbage occasionaly
    local cgi = 0

    -- get first batch
    local Xb = batcher:next()

    -- while batches remain
    while Xb ~= nil do
        -- cuda
        if cuda then
            Xb = Xb:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(Xb)
        local scores_batch = self.model.modules[final_i].output

        if rc_avg then
            -- save the forward orientation
            local preds_batch_fwd = preds_batch:clone()
            local scores_batch_fwd = scores_batch:clone()

            -- reverse complement the sequences
            local Xb_rc = self:rc_seqs(Xb)

            -- predict. preds_batch now holds the reverse
            self.model:forward(Xb_rc)

            -- so add back the forward, and average
            preds_batch = (preds_batch + preds_batch_fwd) / 2
            scores_batch = (scores_batch + scores_batch_fwd) / 2
        end

        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            scores[{pi,{}}] = scores_batch[{i,{}}]:float()
            pi = pi + 1
        end

        -- next batch
        Xb = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    return preds, scores
end


----------------------------------------------------------------
-- predict_finalrep
--
-- Args:
--
-- Predict representations for a new set of sequences.
----------------------------------------------------------------
function ConvNet:predict_finalrepr(Xf, batch_size, Xtens)
    local bs = batch_size or self.batch_size
    local batcher
    if Xtens then
        batcher = BatcherT:__init(Xf, nil, bs)
    else
        batcher = Batcher:__init(Xf, nil, bs)
    end

    -- representation data structure
    local init_repr = true
    local repr

    -- final modules
    local nl_modules = self:get_nonlinears(pool)
    local final_module = nl_modules[#nl_modules]

    local bi = 1

    -- get first batch
    local Xb = batcher:next()

    -- while batches remain
    while Xb ~= nil do
        -- cuda
        if cuda then
            Xb = Xb:cuda()
        end

        -- predict
        self.model:forward(Xb)

        -- get batch repr
        local repr_batch = final_module.output

        -- initialize reprs
        if init_repr then
            repr = torch.FloatTensor(batcher.num_seqs, (#repr_batch)[2])
            init_repr = false
        end

        -- copy into larger tensor
        local pi = bi
        for i = 1,(#repr_batch)[1] do
            repr[{pi,{}}] = repr_batch[{i,{}}]:float()
            pi = pi + 1
        end

        -- next batch
        Xb = batcher:next()
        bi = bi + (#repr_batch)[1]

        collectgarbage()
    end

    return repr
end


----------------------------------------------------------------
-- predict_reprs
--
-- Args:
--  repr_layers
-- Predict representations for a new set of sequences.
----------------------------------------------------------------
function ConvNet:predict_reprs(Xf, batch_size, Xtens, pool, repr_layers)
    local bs = batch_size or self.batch_size
    local batcher
    if Xtens then
        batcher = BatcherT:__init(Xf, nil, bs)
    else
        batcher = Batcher:__init(Xf, nil, bs)
    end

    -- find final model layer
    local final_i = #self.model.modules - 1
    if self.target_type == "continuous" then
        final_i = final_i + 1
    end

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local scores = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local reprs = {}
    local init_reprs = true
    local nl_modules = self:get_nonlinears(pool)
    if repr_layers == nil then
        repr_layers = {}
        for li=1,#nl_modules do
            repr_layers[li] = li
        end
    end

    local bi = 1

    -- get first batch
    local Xb = batcher:next()

    -- while batches remain
    while Xb ~= nil do
        -- cuda
        if cuda then
            Xb = Xb:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(Xb)
        local scores_batch = self.model.modules[final_i].output

        -- copy into larger tensor
        local pi = bi
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            scores[{pi,{}}] = scores_batch[{i,{}}]:float()
            pi = pi + 1
        end

        for li,l in ipairs(repr_layers) do
            -- get batch repr
            local reprs_batch = nl_modules[l].output

            -- initialize reprs
            if init_reprs then
                if reprs_batch:nDimension() == 2 then
                    -- fully connected
                    reprs[l] = torch.FloatTensor(batcher.num_seqs, (#reprs_batch)[2], 1)
                else
                    -- convolution
                    reprs[l] = torch.FloatTensor(batcher.num_seqs, (#reprs_batch)[2], (#reprs_batch)[4])
                end
            end

            -- copy into larger tensor
            local pi = bi
            for i = 1,(#reprs_batch)[1] do
                if reprs_batch:nDimension() == 2 then
                    -- fully connected
                    reprs[l][{pi,{}, 1}] = reprs_batch[{i,{}}]:float()
                else
                    -- convolution
                    reprs[l][{pi,{},{}}] = reprs_batch[{i,{},{}}]:float():squeeze()
                end

                pi = pi + 1
            end
        end

        -- reprs initialized
        init_reprs = false

        -- next batch
        Xb = batcher:next()
        bi = bi + (#preds_batch)[1]

        collectgarbage()
    end

    return preds, scores, reprs
end


----------------------------------------------------------------
-- rc_seqs
--
-- Reverse complement a batch of sequences.
----------------------------------------------------------------
function ConvNet:rc_seqs(seqs)
    local rc_seqs = seqs:clone()
    local seq_len = (#seqs)[4]

    for li=1,seq_len do
        rc_seqs[{{},1,1,li}] = seqs[{{},4,1,seq_len-li+1}]
        rc_seqs[{{},2,1,li}] = seqs[{{},3,1,seq_len-li+1}]
        rc_seqs[{{},3,1,li}] = seqs[{{},2,1,seq_len-li+1}]
        rc_seqs[{{},4,1,li}] = seqs[{{},1,1,seq_len-li+1}]
    end

    return rc_seqs
end

----------------------------------------------------------------
-- sanitize
--
-- Clear the intermediate states in the model before
-- saving to disk.
----------------------------------------------------------------
function ConvNet:sanitize()
    local module_list = self.model:listModules()
    for _,val in ipairs(module_list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if name == 'homeGradBuffers' then val[name] = nil end
            if name == 'input_gpu' then val['input_gpu'] = {} end
            if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
            if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
            if name == 'output' or name == 'gradInput' then
                val[name] = field.new()
            end
            -- batch normalization
            if name == 'buffer' or name == 'normalized' or name == 'centered' then
                val[name] = field.new()
            end
            -- max pooling
            if name == 'indices' then
                val[name] = field.new()
            end
        end
    end
end

----------------------------------------------------------------
-- setStructureParams
--
----------------------------------------------------------------
function ConvNet:setStructureParams(job)
    ---------------------------------------------
    -- training
    ---------------------------------------------
    -- number of examples per weight update
    self.batch_size = job.batch_size or 128

    -- optimization algorithm
    self.optimization = job.optimization or "rmsprop"

    -- base learning rate
    self.learning_rate = job.learning_rate or 0.002

    -- gradient update momentum
    self.momentum = job.momentum or 0.98

    -- adam momentums
    self.beta1 = job.beta1 or 0.9
    self.beta2 = job.beta2 or 0.999

    -- batch normaliztion
    if job.batch_normalize == nil then
        self.batch_normalize = true
    else
        self.batch_normalize = job.batch_normalize
    end

    -- leaky ReLU leak parameter
    -- self.leak = job.leak or 0.01

    -- self.conv_leak = job.conv_leak or self.leak
    -- self.final_leak = job.final_leak or self.leak

    -- normalize weight vectors to this max
    self.weight_norm = job.weight_norm or 10

    ---------------------------------------------
    -- network structure
    ---------------------------------------------
    -- number of filters per layer
    self.conv_filters = job.conv_filters or {10}
    if type(self.conv_filters) == "number" then
        self.conv_filters = {self.conv_filters}
    end

    -- or determine via scaling
    if job.conv_layers and job.conv_filters_scale then
        self.conv_layers = job.conv_layers
        for i=2,self.conv_layers do
            self.conv_filters[i] = math.ceil(job.conv_filters_scale * self.conv_filters[i-1])
        end
    end

    -- determine number of convolution layers
    self.conv_layers = #self.conv_filters

    -- convolution filter sizes
    if job.conv_filter1_size == nil then
        self.conv_filter_sizes = job.conv_filter_sizes or {10}
        if type(self.conv_filter_sizes) == "number" then
            self.conv_filter_sizes = {self.conv_filter_sizes}
        end
    else
        self.conv_filter_sizes = {job.conv_filter1_size}
        local l = 2
        while job[string.format("conv_filter%d_size",l)] ~= nil do
            self.conv_filter_sizes[l] = job[string.format("conv_filter%d_size",l)]
            l = l + 1
        end
    end

    -- or determine via scaling
    if job.conv_layers and job.conv_size_scale then
        for i=2,job.conv_layers do
            self.conv_filter_sizes[i] = math.ceil(job.conv_size_scale * self.conv_filter_sizes[i-1])
        end
    end

    -- convolution filter strides
    self.conv_filter_strides = table_ext(job.conv_filter_strides, 1, self.conv_layers)

    -- pooling widths
    self.pool_width = table_ext(job.pool_width, 1, self.conv_layers)

    -- or determine via scaling
    if job.conv_layers and job.pool_width_scale then
        for i=2,job.conv_layers do
            self.pool_width[i] = math.ceil(job.pool_width_scale * self.pool_width[i-1])
        end
    end

    -- pooling operation ("max", "stochastic")
    self.pool_op = job.pool_op or "max"

    -- random connections (need to test this with one layer, where it becomes irrelevant)
    self.conv_conn = table_ext(job.conv_conn, 1, self.conv_layers-1)


    -- number of hidden units in the final fully connected layers
    self.hidden_units = table_ext(job.hidden_units, 500, 1)

    -- number of fully connected final layers
    self.hidden_layers = #self.hidden_units


    -- target value type ("binary", "continuous", "positive")
    self.target_type = job.target_type or "binary"

    ---------------------------------------------
    -- regularization
    ---------------------------------------------
    -- input dropout probability
    -- self.input_dropout = job.input_dropout or 0

    -- convolution dropout probabilities
    self.conv_dropouts = table_ext(job.conv_dropouts, 0, self.conv_layers)

    -- convolution gaussian noise stdev
    self.conv_gauss = table_ext(job.conv_gauss, 0, self.conv_layers)

    -- final dropout probabilities
    self.hidden_dropouts = table_ext(job.hidden_dropouts, 0, self.hidden_layers)

    -- final gaussian noise stdev
    self.hidden_gauss = table_ext(job.hidden_gauss, 0, self.hidden_layers)

    -- L2 parameter norm
    self.coef_l2 = job.coef_l2 or 0

    -- L1 parameter norm
    self.coef_l1 = job.coef_l1 or 0

end


---------------------------------------------------------------
-- train_epoch
--
-- Train the model for one epoch through the data specified by
-- batcher.
----------------------------------------------------------------
function ConvNet:train_epoch(batcher)
    local total_loss = 0

    -- collect garbage occasionaly
    collectgarbage()
    local cgi = 0

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= self.parameters then
                self.parameters:copy(x)
            end

            -- reset gradients
            self.gradParameters:zero()

            -- evaluate function for mini batch
            local outputs = self.model:forward(inputs)
            local f = self.criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.criterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            -- penalties
            if self.coef_l2 > 0 then
                -- add to loss
                f = f + self.coef_l2 * torch.norm(self.parameters,2)^2/2

                -- add to gradient
                self.gradParameters:add(self.coef_l2, self.parameters)
            end
            if self.coef_l1 > 0 then
                -- add to loss
                f = f + self.coef_l1 * torch.norm(self.parameters,1)

                -- add to gradient
                self.gradParameters:add(torch.sign(self.parameters):mul(self.coef_l1))
            end

            -- return f and df/dX
            return f, self.gradParameters
        end

        -- perform RMSprop step
        if self.optimization == "rmsprop" then
            self.optim_state = self.optim_state or {
                learningRate = self.learning_rate,
                alpha = self.momentum
            }
            optim.rmsprop(feval, self.parameters, self.optim_state)

        elseif self.optimization == "adam" then
            self.optim_state = self.optim_state or {
                learningRate = self.learning_rate,
                beta1 = self.beta1,
                beta2 = self.beta2
            }
            optim.adam(feval, self.parameters, self.optim_state)
        else
            print("Unrecognized optimization algorithm")
            exit(1)
        end

        -- cap weight paramaters
        self.model:maxParamNorm(self.weight_norm)

        -- accumulate loss
        total_loss = total_loss + self.criterion.output

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- reset batcher
    batcher:reset()

    -- mean loss over examples
    avg_loss = total_loss / batcher.num_seqs
    return avg_loss
end


----------------------------------------------------------------
-- test
--
-- Predict targets for X and compare to Y.
----------------------------------------------------------------
function ConvNet:test(Xf, Yf, batch_size, rc_avg)
    -- track the loss across batches
    local loss = 0

    -- collect garbage occasionaly
    local cgi = 0

    -- create a batcher to help
    local batch_size = batch_size or self.batch_size
    local batcher = Batcher:__init(Xf, Yf, batch_size)

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(inputs)

        if rc_avg then
            -- save the forward orientation
            local preds_batch_fwd = preds_batch:clone()

            -- reverse complement the sequences
            local rc_inputs = self:rc_seqs(inputs)

            -- predict. preds_batch now holds the reverse
            self.model:forward(rc_inputs)

            -- so add back the forward, and average
            preds_batch = (preds_batch + preds_batch_fwd) / 2
        end

        -- accumulate loss
        loss = loss + self.criterion:forward(preds_batch, targets)

        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            pi = pi + 1
        end

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- mean loss over examples
    local avg_loss = loss / batcher.num_seqs

    -- save pred means and stds
    self.pred_means = preds:mean(1):squeeze()
    self.pred_stds = preds:std(1):squeeze()

    local Ydim = batcher.num_targets
    if self.target_type == "binary" then
        -- compute AUC
        local AUCs = torch.Tensor(Ydim)
        local roc_points = {}
        for yi = 1,Ydim do
            -- read Yi from file
            local Yi = Yf:partial({1,batcher.num_seqs},{yi,yi}):squeeze()

            if(Yi:sum() == 0) then
                roc_points[yi] = nil
                AUCs[yi] = 1.0

            else
                -- compute ROC points
                roc_points[yi] = ROC.points(preds[{{},yi}], Yi)

                -- compute AUCs
                AUCs[yi] = ROC.area(roc_points[yi])
            end

            collectgarbage()
        end

        return avg_loss, AUCs, roc_points
    else
        -- compute R2
        local R2s = variance_explained(Yf, preds)
        local cors = spearmanr(Yf, preds)

        return avg_loss, R2s, cors
    end
end


----------------------------------------------------------------
-- test_mc
--
-- Predict targets for X and compare to Y.
----------------------------------------------------------------
function ConvNet:test_mc(Xf, Yf, mc_n, batch_size)
    -- requires stochasticity
    self.model:training()

    -- track the loss across batches
    local loss = 0

    -- collect garbage occasionaly
    local cgi = 0

    -- create a batcher to help
    local batch_size = batch_size or self.batch_size
    local batcher = Batcher:__init(Xf, Yf, batch_size)

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(inputs)
        for mi = 2,mc_n do
            local preds_batch_mc = self.model:forward(inputs)
            preds_batch = preds_batch + preds_batch_mc
        end
        preds_batch = preds_batch / mc_n

        -- accumulate loss
        loss = loss + self.criterion:forward(preds_batch, targets)

        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            pi = pi + 1
        end

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- mean loss over examples
    local avg_loss = loss / batcher.num_seqs

    -- save pred means and stds
    self.pred_means = preds:mean(1):squeeze()
    self.pred_stds = preds:std(1):squeeze()

    local Ydim = batcher.num_targets
    if self.target_type == "binary" then
        -- compute AUC
        local AUCs = torch.Tensor(Ydim)
        local roc_points = {}
        for yi = 1,Ydim do
            -- read Yi from file
            local Yi = Yf:partial({1,batcher.num_seqs},{yi,yi}):squeeze()

            if(Yi:sum() == 0) then
                roc_points[yi] = nil
                AUCs[yi] = 1.0

            else
                -- compute ROC points
                roc_points[yi] = ROC.points(preds[{{},yi}], Yi)

                -- compute AUCs
                AUCs[yi] = ROC.area(roc_points[yi])
            end

            collectgarbage()
        end

        return avg_loss, AUCs, roc_points
    else
        -- compute R2
        local R2s = variance_explained(Yf, preds)

        return avg_loss, R2s
    end
end

----------------------------------------------------------------
-- table_ext
--
-- Extend the table to be the given size, adding in the default
-- value to fill it.
----------------------------------------------------------------
function table_ext(try, default, size)
    -- set var to try if available or default otherwise
    var = try or default

    -- if it was a number
    if type(var) == "number" then
        -- change default to the number
        default = var

        -- make it a table
        var = {var}
    end

    -- extend the table if too small
    for i = 2,size do
        if i > #var then
            var[i] = default
        end
    end

    return var
end
