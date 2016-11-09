local Model = {
  nFilters = 16
}

--[[
SpatialConvolution
owidth  = floor((width  + 2*padW - kW) / dW + 1)
oheight = floor((height + 2*padH - kH) / dH + 1)
]]--

require 'Probe'
function Model:CreateRecogniser(input_size, hidden_size, x_size, w_size, number_of_mixtures)
  local height = input_size[1] -- 28x28 for mnist
  local width = input_size[2]
  local nChannels = self.nChannels
  local nFilters = self.nFilters

  local input = - nn.Identity()
	local hidden = input
          - nn.View(-1, nChannels, height, width)
					- nn.SpatialConvolution(nChannels, nFilters, 6,6,1,1,0,0)
					- nn.SpatialBatchNormalization(nFilters)
					- nn.ReLU(true)
          - nn.SpatialConvolution(nFilters, 2*nFilters, 6,6,1,1,0,0)
          - nn.SpatialBatchNormalization( 2*nFilters)
					- nn.ReLU(true)
          - nn.SpatialConvolution(2*nFilters, 4*nFilters, 4,4,2,2,1,1)
          - nn.SpatialBatchNormalization( 4*nFilters)
          - nn.ReLU(true)
          - nn.SpatialConvolution(4*nFilters, hidden_size, 9,9)
          - nn.SpatialBatchNormalization( hidden_size)
          - nn.ReLU(true)
          --out: hidden x 1 x 1
          - nn.View(-1, hidden_size)

	local q_z = hidden
				- nn.Linear(hidden_size, number_of_mixtures)
				- nn.SoftMax()


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

	return nn.gModule({input}, {q_z, q_x, q_w})
end

--[[
SpatialFullConvolution
owidth  = (width  - 1) * dW - 2*padW + kW + adjW
oheight = (height - 1) * dH - 2*padH + kH + adjH
]]--

function Model:CreateYGenerator(input_size, hidden_size, output_size, continous)
  local height = output_size[1]
  local width = output_size[2]
  local nChannels = self.nChannels
  local nFilters = self.nFilters

	local input = - nn.Identity()
	local hidden = input
          - nn.Linear(input_size, hidden_size)
          - nn.BatchNormalization(hidden_size)
          - nn.ReLU(true)
          - nn.View(-1, hidden_size, 1, 1)
          - nn.SpatialFullConvolution(hidden_size, 4*nFilters, 9,9)
          - nn.SpatialBatchNormalization(4*nFilters)
          - nn.ReLU(true)
          - nn.SpatialFullConvolution(4*nFilters, 2*nFilters, 4,4,2,2,1,1)
          - nn.SpatialBatchNormalization( 2*nFilters)
          - nn.ReLU(true)
          - nn.SpatialFullConvolution(2*nFilters, nFilters, 6,6,1,1,0,0)
          - nn.SpatialBatchNormalization( nFilters)
          - nn.ReLU(true)
          - nn.SpatialFullConvolution( nFilters, nChannels, 6,6,1,1,0,0)
          - nn.View(-1, nChannels, height, width)

	local output = hidden - nn.Sigmoid(true)

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
