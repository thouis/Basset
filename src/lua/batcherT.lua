BatcherT = {}
BatcherT.__index = BatcherT

function BatcherT:__init(X, Y, batch_size, permute)
    bat = {}
    setmetatable(bat, self)

    bat.X = X
    bat.Y = Y

    bat.num_seqs = (#X)[1]

    bat.batch_size = batch_size
    bat.permute = permute or false

    bat:reset()

    return bat
end

function BatcherT:next()
    local X_batch = nil
    local Y_batch = nil

    if self.start <= (#self.X)[1] then
        -- allocate Tensors
        local blen = self.stop-self.start+1
        X_batch = torch.Tensor(blen, 4, 1, (#self.X)[4])
        if self.Y ~= nil then
            Y_batch = torch.Tensor(blen, (#self.Y)[2])
        end

        -- copy data
        local k = 1
        for i = self.start, self.stop do
            X_batch[k] = self.X[self.order[i]]
            if self.Y ~= nil then
                Y_batch[k] = self.Y[self.order[i]]
            end
            k = k + 1
        end

        -- update batch indexes for next
        self.start = self.start + self.batch_size
        self.stop = math.min(self.stop + self.batch_size, (#self.X)[1])
    end

    return X_batch, Y_batch
end

function BatcherT:reset()
    self.start = 1
    self.stop = math.min(self.start+self.batch_size-1, (#self.X)[1])

    if self.permute then
        self.order = torch.randperm((#self.X)[1])
    else
        self.order = {}
        for i = 1,(#self.X)[1] do
            self.order[i] = i
        end
    end
end