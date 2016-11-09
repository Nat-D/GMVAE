
--require 'Probe'

function GaussianSampler(M, D)

    local q = - nn.Identity()
    local eps = - nn.Identity()
    local mean = q  - nn.SelectTable(1)
                    - nn.Replicate(M)
    local logVar = q - nn.SelectTable(2)
    local std   = logVar - nn.MulConstant(0.5) 
                         - nn.Exp()
                         - nn.Replicate(M)
    local noise =  {std, eps} - nn.CMulTable()
    local sample = {mean, noise}
                         - nn.CAddTable() -- [MxNxD]
                         - nn.View(-1, D) -- [(MxN)xD]
    return nn.gModule({q, eps}, {sample})
end


-- TEST PASS
function KLDivergence(D, M)
    -- KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
    local mean1_in = - nn.Identity()
    local logVar1_in = - nn.Identity()
    local mean2_in = - nn.Identity() -- [(MxN)xD]
    local logVar2_in = - nn.Identity() -- [(MxN)xD]

    local mean1 = mean1_in - nn.Replicate(M)
    local logVar1 = logVar1_in - nn.Replicate(M)
    local mean2 = mean2_in - nn.View(M, -1, D)
    local logVar2 = logVar2_in - nn.View(M, -1, D)

    local var1 = logVar1 - nn.Exp()
    local var2 = logVar2 - nn.Exp()
    local dm2 = {mean1, mean2}
                - nn.CSubTable()
                - nn.Power(2)
    local dm2_v1 = {dm2, var1} - nn.CAddTable()
    local dm2_v1_v2 = {dm2_v1, var2} - nn.CDivTable() - nn.AddConstant(-1)
    local total = {dm2_v1_v2, logVar2} - nn.CAddTable()
    local totals = {total, logVar1}
                    - nn.CSubTable()
                    - nn.MulConstant(0.5) -- [MxNxD]
                    - nn.Sum(1)
                    - nn.MulConstant(1/M)
                    - nn.View(-1, D, 1)

    return nn.gModule({mean1_in, logVar1_in, mean2_in, logVar2_in}, {totals})

end

function KL_Table(K, D, M)
    local KL_table = nn.ConcatTable()
    for k=1, K do
        local mean   = - nn.Identity() -- [NxD]
        local logVar = - nn.Identity() -- [NxD]
        local mean_Mixture = - nn.Identity() -- {[NxD]}k
        local logVar_Mixture = - nn.Identity() -- {[NxD]}k

        local meanK = mean_Mixture - nn.SelectTable(k)
        local logVarK = logVar_Mixture - nn.SelectTable(k)
        local KL = {mean, logVar, meanK, logVarK} - KLDivergence(D, M)
        local KL_module = nn.gModule({mean, logVar, mean_Mixture, logVar_Mixture}, {KL})
        KL_table:add(KL_module)
    end

    return KL_table
end

function ExpectedKLDivergence(K, D, M)

    local q_z    = - nn.Identity() -- [NxK]
    local mean   = - nn.Identity() -- [NxD]
    local logVar = - nn.Identity() -- [NxD]
    local mean_Mixture = - nn.Identity() -- {[NxD]}k
    local logVar_Mixture = - nn.Identity() -- {[NxD]}k

    local KL_List = {mean, logVar, mean_Mixture, logVar_Mixture}
                - KL_Table(K, D, M)  -- {[NxDx1]}k
                - nn.JoinTable(3) -- [NxDxK]

    local weighted_KL = {KL_List, q_z}
                    - nn.MV()  -- [NxDxK]x[NxK] = [NxD]
    return  nn.gModule({q_z, mean,logVar, mean_Mixture, logVar_Mixture},{weighted_KL})
end




require 'layers/GaussianLogLikelihood'
function Likelihood(K, D, M)
    local x_sample = - nn.Identity() -- [(MxN)xD]
    local mean = - nn.Identity()  -- {[(MxN)xD]}k
    local logVar = - nn.Identity() -- {[(MxN)xD]}k

    local llh_table = nn.ConcatTable()
    for k =1, K do
        local x = - nn.Identity()
        local mean_k_in = - nn.Identity()
        local logVar_k_in = - nn.Identity()

        local mean_k = mean_k_in
                        - nn.SelectTable(k)

        local logVar_k = logVar_k_in
                        - nn.SelectTable(k)

        local llh = {x, mean_k, logVar_k}
                        - nn.GaussianLogLikelihood()

        local llh_module = nn.gModule({x, mean_k_in, logVar_k_in}, {llh})
        llh_table:add(llh_module)
    end

    local out = {x_sample, mean, logVar}
                            - llh_table -- {[MxN,1]}k
                            - nn.JoinTable(2) -- [MxN,K]  -- log unNorm P
                            - nn.SoftMax()

    return nn.gModule({x_sample, mean, logVar},{out})
end
