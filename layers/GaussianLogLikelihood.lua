

local GaussianLogLikelihood, parent = torch.class('nn.GaussianLogLikelihood', 'nn.Module')

function GaussianLogLikelihood:__init(name,display)
  parent.__init(self)
  self.gradInput = {}
end

function GaussianLogLikelihood:evaluate()
  self._x = nil
  self._var = nil
  self.output = nil
end

function GaussianLogLikelihood:updateOutput(input)
  -- input[1] : x [NxD]
  -- input[2] : mean [NxD]
  -- input[3] : logVar [NxD]
  -- llh = -0.5 sum_d { (x_i - mu_i)^2/var_i } - 1/2 sum_d (logVar_i) - D/2 ln(2pi) [N]
  local N = input[1]:size(1)
  local D = input[1]:size(2)

  self._x = self._x or torch.Tensor():typeAs(input[1]):resizeAs(input[1])
  self._x:copy(input[1])
  self._var = self._var or torch.Tensor():typeAs(input[3]):resizeAs(input[3])
  self._var:copy(input[3])
  self._var:exp()

  self.output = self.output or input[1].new()
  self.output:typeAs(input[1]):resize(N, 1):zero()



  self.output:copy( self._x:add(-1, input[2]):pow(2):cdiv(self._var):sum(2) )
  self.output:add( input[3]:sum(2) )
  self.output:add( D * torch.log(2*math.pi) )
  self.output:mul(-0.5)

  return self.output
end

function GaussianLogLikelihood:updateGradInput(input, gradOutput)
  -- dL_dllh = gradOutput [N]
  -- dL_dx_i = dL_dllh * dllh_dx_i = gradOutput * ( - (x_i - mu_i)/var_i)  )
  -- dL_dmu_i = gradOutput * ( (x_i - mu_i)/var_i )
  -- dL_dlogVar_i = gradOutput * dllh_dlogVar_i = gradOutput * (  0.5 * (x_i - mu_i)^2 *(1/Var) - 0.5  )

  -- 1/var_i = exp( -log Var_i)
  local x_m = torch.add(input[1], -1, input[2])
  self.gradInput[1] = torch.exp(-input[3]):cmul( x_m ):cmul(gradOutput:expandAs(x_m)):mul(-1)
  self.gradInput[2] = self.gradInput[1]:clone():mul(-1)
  self.gradInput[3] = torch.exp(-input[3]):cmul(x_m):cmul(x_m):mul(0.5):add(-0.5):cmul(gradOutput:expandAs(x_m))

  return self.gradInput
end
