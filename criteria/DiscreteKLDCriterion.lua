
local DiscreteKLDCriterion, parent = torch.class('nn.DiscreteKLDCriterion', 'nn.Criterion')

function DiscreteKLDCriterion:__init(weight)
  self.weight = weight

end

function DiscreteKLDCriterion:updateOutput(input)
   	-- KL(q(z)||p(z)) =  - sum_k q(k) log p(k)/q(k)
   	-- let's p(k) = 1/K

   local KLelement = torch.Tensor():typeAs(input):resize(input:size(1)):zero()
   local input_t = input:t()
   local K = input:size(2)

   for k = 1, K do
   	KLelement:add( torch.cmul( input_t[k], ( torch.log( K * input_t[k] + 1e-10 ) ) ) )
   end
   self.output = KLelement:sum()
    return self.output * self.weight
end

function DiscreteKLDCriterion:updateGradInput(input)

	local K = input:size(2)

	self.gradInput = self.gradInput or input.new()
	self.gradInput:typeAs(input):resizeAs(input)
	-- dKL_dqk = - q(k) * ( -1/q(k)  ) + (- log p(k) + log q(k))
	-- dKL_dqk =  (1 + log q(k) - logP(k) )

	self.gradInput:copy(input):div(K):add(1e-10):log():add(1):mul(self.weight)
    return self.gradInput
end
