import matplotlib.pyplot as plt

loss_document=open('IWSLT_2016/english_french/models/NLL_COS_BPE_16000_neubig_style_training/loss_model_2.txt','r')

list_nll_loss=[]
list_cos_loss=[]
list_nll_cos_loss=[]

for line in loss_document:
    nll_loss=float(line.split('\t')[0])
    cos_loss = float(line.split('\t')[1])
    list_nll_loss.append(nll_loss)
    list_cos_loss.append(cos_loss)
    list_nll_cos_loss.append(nll_loss)

plt.plot(list_nll_cos_loss)
plt.ylabel('nll_loss')
plt.show()