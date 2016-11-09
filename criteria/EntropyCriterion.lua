
local EntropyCriterion, parent = torch.class('nn.EntropyCriterion', 'nn.Criterion')

function EntropyCriterion:__init(weight, nMC)
  self.weight = weight or 1
  self.nMC = nMC
end

function EntropyCriterion:updateOutput(input)
  local N = input:size(1)
  -- H(Z|X) = E_q(x)E_p(z|x)[- log P(z|x)]
  self.output = torch.log(input + 1e-10):cmul(input):sum()

  return - (self.output/ self.nMC) --* self.weight
end

function EntropyCriterion:updateGradInput(input)
  -- input [N, K]
  -- dH_dp_ik = d_dp_ik ( p_k[-logP_k] ) = - 1/N( 1 + log p_k )
  --local N = input:size(1)

  self.gradInput = self.gradInput or input.new()
  self.gradInput:typeAs(input):resizeAs(input)
  self.gradInput:copy(input + 1e-10):log():add(1):mul(-1)

  return self.gradInput:mul(self.weight):div(self.nMC)

end
