

local Model = {}

function Model:CreateRecogniser(input_size, hidden_size, x_size, w_size, number_of_mixtures)
	local input = - nn.Identity()
	local hidden = input
					- nn.Linear(input_size, hidden_size)
					- nn.BatchNormalization(hidden_size)
					- nn.ReLU(true)
					- nn.Linear(hidden_size, hidden_size)
					- nn.BatchNormalization(hidden_size)
					- nn.ReLU(true)

	--[[
	local q_z = hidden
				- nn.Linear(hidden_size, number_of_mixtures)
				- nn.SoftMax()
	]]--

	local mean_x = hidden
				- nn.Linear(hidden_size, x_size)
	local logVar_x = hidden
				- nn.Linear(hidden_size, x_size)
	local mean_w = hidden
				- nn.Linear(hidden_size, w_size)
	local logVar_w = hidden
				- nn.Linear(hidden_size, w_size)

	local q_x = {mean_x, logVar_x} - nn.Identity()
	local q_w = {mean_w, logVar_w} - nn.Identity()

	return nn.gModule({input}, {q_x, q_w})
end

function Model:CreateYGenerator(input_size, hidden_size, output_size, continous)
	local input = - nn.Identity()
	local hidden = input
					- nn.Linear(input_size, hidden_size)
					- nn.ReLU(true)
					- nn.Linear(hidden_size, hidden_size)
					- nn.ReLU(true)


	local out = nn.Sequential()
	if continous == 1 then
		local table = nn.ConcatTable()
		table:add(nn.Linear( hidden_size, output_size) )
		table:add(nn.Linear( hidden_size, output_size) )
		out:add(table)
	else
		out:add(nn.Linear( hidden_size, output_size))
		out:add(nn.Sigmoid(true))
	end

	local output = hidden - out

	return nn.gModule({input}, {output})
end

function Model:CreatePriorGenerator(input_size, hidden_size, output_size, number_of_mixtures)

	local input = - nn.Identity()
	local hidden = input
					- nn.Linear(input_size, hidden_size)
					- nn.Tanh(true)

	local outTable = nn.ConcatTable()
	for k=1, number_of_mixtures do
		outTable:add(nn.Linear(hidden_size, output_size))
	end
	local outTable2 = nn.ConcatTable()
	for k=1, number_of_mixtures do
		outTable2:add(nn.Linear(hidden_size, output_size))
	end

	local mean 	= hidden
					- outTable

	local logVar  = hidden
					- outTable2


	return	nn.gModule({input}, {mean, logVar})
end

return Model
