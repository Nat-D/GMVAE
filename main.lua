require 'nn'
require 'optim'
require 'nngraph'
require 'ClusteringEvaluations'
local gnuplot = require 'gnuplot'
local cjson = require 'cjson'
local image = require 'image'

local cmd = torch.CmdLine()
cmd:option('-dataSet', 'svhn', 'Dataset used')
cmd:option('-seed', 1, 'random seed')
cmd:option('-learningRate', 0.0001, 'Learning rate')
cmd:option('-batchSize', 128, 'Batch Size')
cmd:option('-optimiser', 'adam', 'Optimser')
cmd:option('-gpu', 1, 'Using Cuda, 1 to enable')
cmd:option('-epoch', 500, 'Number of Epoch')
cmd:option('-visualise2D', 0, 'Save data for visualisation, 1 to enable')
cmd:option('-xSize', 200, 'Size of x variable')
cmd:option('-wSize', 150, 'Size of w variable')
cmd:option('-K', 10, 'Number of clusters')
cmd:option('-hiddenSize', 500, 'Size of the hidden layer')
cmd:option('-zPriorWeight', 1, 'Weight on the z prior term')
cmd:option('-ACC', 1, 'Report Clustering accuracy')
cmd:option('-visualGen', 1, 'Visualise the generation samples at every [input] epochs (0 to disable)')
cmd:option('-continuous', 0, 'Data is continous use Gaussian Criterion (1), Data is discrete use BCE Criterion (0)')
cmd:option('-labelRatio', 0, 'Ratio between supervised signal and total data')
cmd:option('-inputDimension', 1, 'Dimension of the input vector into the network (e.g. 2 for height x width)')
cmd:option('-network', 'conv', 'Network architecture use: fc (FullyConnected)/ conv (Convolutional AE)/ hconv (half convolution)/ resnet (residual network)')
cmd:option('-nChannels', 3, 'Number of Input channels')
cmd:option('-nFilters', 64, 'Number of Convolutional Filters in first layer')
cmd:option('-preprocess', 1, 'Preprocess the dataset (0 to disable)')
cmd:option('-nMC', 1, 'Number of monte-carlo sample')
cmd:option('-gpuID', 1, 'Set GPU id to use')
cmd:option('-cvWeight', 1, 'Weight of the information theoretic cost term')
cmd:option('-_id', '', 'Experiment Path')
cmd:option('-saveModel',0, 'Save model in experiments folder after finish training')
cmd:option('-reportLowerBound', 1, 'Save lowerbound while training')
cmd:option('-subEpoch', 50000, 'How many data per sub epoch?')
cmd:option('-numTest', 26000, 'How many test data to use?')
cmd:option('-saveFig', 20, 'How frequent we save the figure?')
local opt = cmd:parse(arg)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Set save directory
if not paths.dirp('experiments') then
	paths.mkdir('experiments')
end
paths.mkdir(paths.concat('experiments', opt._id))
-- Save option for reference
local file = torch.DiskFile(paths.concat('experiments', opt._id, 'opts.json'),'w')
file:writeString(cjson.encode(opt))
file:close()

-- Preparing Data --
local dataSet = opt.dataSet-- 'spiral'
local data, label_data, width, height, label_key_shuffled
local y_size

print('Prepare data')

svhn = require 'datasets/svhn'
data_all = svhn.trainData.data:float():div(255)
extra_data_all = svhn.extraTrainData.data:float():div(255)
data_all = torch.cat(data_all, extra_data_all, 1)
extra_data_all = nil
label_data_all = svhn.trainData.labels
extra_label_data = svhn.extraTrainData.labels
label_data_all = torch.cat(label_data_all, extra_label_data, 1)
extra_label_data = nil
test_data = svhn.testData.data[{{1,opt.numTest}}]:float():div(255) -- 26032 might want to throw away 32
test_data_label = svhn.testData.labels[{{1,opt.numTest}}]:float()



width = 32
height = 32
y_size = {32,32}
--width = 96
--height = 96
--y_size = {96,96}
local N_all = data_all:size(1)



local x_size = opt.xSize
local z_size = opt.K
local w_size = opt.wSize
local batch_size = opt.batchSize
local learning_rate = opt.learningRate
local optimiser = opt.optimiser
local max_epoch = opt.epoch
local hidden_size = opt.hiddenSize
local cuda = opt.gpu
local visualise2D = opt.visualise2D
local visualGen = opt.visualGen
local zPriorWeight = opt.zPriorWeight
local reportACC = opt.ACC
local continuous = opt.continuous

require 'UtilsNetwork'
local model

model = require 'models/SVHNConvAE'
model.nChannels = opt.nChannels
model.nFilters = opt.nFilters

-- Network parts --
local zxw_recogniser = model:CreateRecogniser(y_size, hidden_size, x_size, w_size, z_size)
local y_generator = model:CreateYGenerator(x_size, hidden_size, y_size, continuous)
local prior_generator = model:CreatePriorGenerator(w_size, hidden_size, x_size, z_size)

-- Connect parts together --
local input = - nn.Identity()
local noise1 = - nn.Identity()
local noise2 = - nn.Identity()
local zxw = input - zxw_recogniser
local q_x = zxw - nn.SelectTable(1)
local q_w = zxw - nn.SelectTable(2)
local x_sample = {q_x, noise1} - GaussianSampler(opt.nMC, x_size)
local w_sample = {q_w, noise2} - GaussianSampler(opt.nMC, w_size)
local y_recon  = x_sample - y_generator
local p_xz  = w_sample - prior_generator
local mean_k = p_xz - nn.SelectTable(1)
local logVar_k = p_xz - nn.SelectTable(2)
local p_z = {x_sample, mean_k, logVar_k} - Likelihood(z_size, x_size, opt.nMC)
local GMVAE = nn.gModule({input, noise1, noise2},{p_z, q_x, q_w, p_xz, y_recon})

--
local MC_replicate = nn.Replicate(opt.nMC)
--

require 'criteria/GaussianCriterion'
require 'criteria/VAE_KLDCriterion'
require 'criteria/DiscreteKLDCriterion_with_Target'
require 'criteria/EntropyCriterion'
require 'criteria/DiscreteKLDCriterion'
-- Criteria --
local ReconCriterion
if continuous == 1 then
	ReconCriterion = nn.GaussianCriterion()
else
	ReconCriterion = nn.BCECriterion()
	ReconCriterion.sizeAverage = false
end

local ExpectedKLDivergence = ExpectedKLDivergence(z_size, x_size, opt.nMC)
local VAE_KLDCriterion = nn.VAE_KLDCriterion()
local DiscreteKLDCriterion = nn.DiscreteKLDCriterion(zPriorWeight)
local EntropyCriterion = nn.EntropyCriterion(opt.cvWeight, opt.nMC)

-- Set up Cuda
if cuda == 1 then
	require 'cutorch'
	require 'cunn'
	print('Using Cuda')
	local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
	cutorch.setDevice(opt.gpuID)

	GMVAE:cuda()
	ReconCriterion:cuda()
	ExpectedKLDivergence:cuda()
	VAE_KLDCriterion:cuda()
	DiscreteKLDCriterion:cuda()

	MC_replicate:cuda()
	EntropyCriterion:cuda()

	if hasCudnn then
	  print('Using cudnn')
   	  cudnn.convert(GMVAE, cudnn)
  	end
	test_data = test_data:cuda()
end

local params, gradParams = GMVAE:getParameters()
local oneTensor

function feval(params)

	GMVAE:zeroGradParameters()

	local qZ_no_mask, qX, qW, p_xz, y_recon, x_sample  = table.unpack( GMVAE:forward({y, n1, n2}) )

	-- CALCULATE LOSS AND GRADIENT
 	local y_replicated = MC_replicate:forward(y)

	-- 1.) Reconstruction Cost = -E[logP(y|x)]
	local reconLoss = ReconCriterion:forward(y_recon, y_replicated  ) * (1/opt.nMC)
	local dRecon_dy = ReconCriterion:backward(y_recon, y_replicated )

	if continuous == 1 then
		dRecon_dy[1]:div(opt.nMC)
		dRecon_dy[2]:div(opt.nMC)
	else
		dRecon_dy:div(opt.nMC)
	end


	-- 2.) E_z_w[KL(q(x)|| p(x|z,w))]
	local qZ = qZ_no_mask
	local mean_x, logVar_x = table.unpack(qX)
	local mean_k, logVar_k = table.unpack(p_xz)
	local KL_out = ExpectedKLDivergence:forward({qZ, mean_x, logVar_x, mean_k, logVar_k})
	local xLoss = KL_out:sum()
	oneTensor = oneTensor or torch.Tensor():typeAs(KL_out):resizeAs(KL_out):fill(1)
	local gradQz_no_mask, gradMean_x, gradLogVar_x, gradMean_k, gradLogVar_k =
			table.unpack( ExpectedKLDivergence:backward({qZ, mean_x, logVar_x, mean_k, logVar_k}, oneTensor ) )


	-- 3.) KL( q(w) || P(w) )
	local mean_w, logVar_w = table.unpack(qW)
	local wLoss = VAE_KLDCriterion:forward( mean_w, logVar_w)
	local gradW	= VAE_KLDCriterion:backward( mean_w, logVar_w)

	-- 4.) KL( q(z) || P(z) )  : P(z) can be shaped with known label
	local zLoss = 0
	zLoss = DiscreteKLDCriterion:forward(qZ_no_mask)
	gradQz_no_mask:add( DiscreteKLDCriterion:backward(qZ_no_mask) )

	-- 5.) we want to add regularizer based on consistency violation (CV):

	local lh = Likelihood:forward({x_sample, mean_k, logVar_k})
	local CV = EntropyCriterion:forward(lh)
	local gradLh = EntropyCriterion:backward(lh)
	local gradX, gradM_k, gradLv = table.unpack(Likelihood:backward({x_sample, mean_k, logVar_k}, gradLh))

	for k =1, z_size do
		gradMean_k[k]:add(gradM_k[k])
		gradLogVar_k[k]:add(gradLv[k])
	end

	-- Put all the gradient of the cost into a table
	local gradLoss = { gradQz_no_mask,
					   { gradMean_x, gradLogVar_x},
					   gradW,
					   { gradMean_k, gradLogVar_k  },
					   dRecon_dy,
					   gradX
					 }


	GMVAE:backward({y, n1, n2}, gradLoss)

	local loss = reconLoss + xLoss + wLoss + zLoss
	return {loss,CV}, gradParams
end



local data_split = data_all:split(opt.subEpoch)
local label_split= label_data_all:split(opt.subEpoch)

local max_sub_epoch = table.getn(data_split)
local ACC_evaluation = torch.Tensor(max_epoch*max_sub_epoch):zero()
local Class_evaluation = torch.Tensor(max_epoch*max_sub_epoch):zero()
local Train_Class_evaluation = torch.Tensor(max_epoch*max_sub_epoch):zero()
local Train_ACC_evaluation = torch.Tensor(max_epoch*max_sub_epoch):zero()
local Lowerbound = torch.Tensor(max_epoch*max_sub_epoch):zero()
local CV_Cost = torch.Tensor(max_epoch*max_sub_epoch):zero()




for epoch = 1, max_epoch do

	for sub_epoch = 1, table.getn(data_split) do
		-- load to gpu
		local data = data_split[sub_epoch]:cuda()
		local label_data = label_split[sub_epoch]:float()

		local N = data:size(1)

		local Noise1 = torch.randn(opt.nMC, N, x_size)
		local Noise2 = torch.randn(opt.nMC, N, w_size)

		if cuda == 1 then
			Noise1 = torch.CudaTensor():resize(opt.nMC, N, x_size):copy(Noise1)
			Noise2 = torch.CudaTensor():resize(opt.nMC, N, w_size):copy(Noise2)
		end

		local indices_all = torch.randperm(N):long()
		local indices = indices_all:split(batch_size)
		indices[#indices] = nil

		local recon, latent, w_latent, labels

		if reportACC == 1 then
			labels = torch.Tensor():resize(N,z_size):zero()
		end

		local Loss = 0.0
		local CVLoss = 0.0

		--============================================--

		for t,v in ipairs(indices) do

			GMVAE:training()
			xlua.progress(t, #indices)
			y = data:index(1,v)
			n1 = Noise1:index(2,v)
			n2 = Noise2:index(2,v)

			__, loss = optim[optimiser](feval, params, {learningRate = learning_rate })

			Loss = Loss + loss[1][1]
			CVLoss = CVLoss + loss[1][2]

		end
		print("Epoch: " .. epoch .." SubEpoch: "..sub_epoch.."/"..table.getn(data_split).. " Loss: " .. Loss/N )

		collectgarbage()

		if visualGen > 0 then
			-- Generate data condition on the code every 'visualGen' epoch
			if epoch%visualGen == 0 then
				GMVAE:evaluate()
				-- : Fix w = 0 and vary z from 1 to K,
				-- : Samples 10 x
				local gaussianSampler = GaussianSampler(1, x_size)
				if cuda == 1 then
					gaussianSampler:cuda()
				end
				local samples = torch.Tensor(opt.nChannels, z_size * height, 10 * width):typeAs(data)
				local mean, logVar = table.unpack( prior_generator:forward(torch.Tensor(10, w_size):typeAs(data):zero()) )
				for k=1, z_size do
					local eps = torch.randn(1, mean[k]:size(1), mean[k]:size(2)):typeAs(mean[k])
					local sample_x = gaussianSampler:forward({ {mean[k], logVar[k]}, eps})
					local samples_temp = y_generator:forward(sample_x):view(-1, opt.nChannels, width, height)
					for j = 1, 10 do
						samples[{{}, {(k-1) * height + 1, k * height},{(j-1)*width + 1, j*width}}] = samples_temp[j]
					end
				end
				image.save(paths.concat('experiments', opt._id, 'samples.png'), samples)
				if epoch%opt.saveFig == 0 then
					image.save(paths.concat('experiments', opt._id, 'samples'..epoch..'.png'), samples)
				end
			end
		end

	end

end
