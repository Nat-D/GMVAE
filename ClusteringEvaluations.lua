
function AAE_Clustering_Criteria(generated_label, true_label)
  --[[ Clustering Criteria of the Adversarial Autoencoder
      "Once the training is done, for each cluster i,
       we found the validation example x_n that maximizes q(y_i|x_n)
        and assigned the label of x_n to all points in cluster i"
  ]]--
  -- generated_label [NxK]
  -- true_label [N]
  local K = generated_label:size(2)
  local __, x_n = generated_label:max(1)
  local __, labels = generated_label:max(2)

  local ACC = 0
  for k =1, K do
     -- assign cluster k as label true_label[x_n[k]]
    ACC = ACC + ( labels:eq(k) +  true_label:eq(true_label[x_n[1][k]]) ):eq(2):sum()
  end
  ACC = ACC/generated_label:size(1)

  return ACC
end

function Classification_Score(generated_label, true_label)
    local __, labels = generated_label:max(2)

    return labels:float():eq(true_label):sum()/generated_label:size(1)
end
