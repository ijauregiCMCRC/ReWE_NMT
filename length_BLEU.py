import os

refs_file=open('IWSLT_2016/en-fr_test_2015_2016/test_2015_plus_2016.tok.fr')
seed=3
hyps_BASELINE_file=open('IWSLT_2016/english_french/models/BASELINE_PRE_EMBS_neubig_style_training/SEED_models/seed_'+str(seed)+'/pred_5_test.txt')
hyps_NLL_COS_file=open('IWSLT_2016/english_french/models/NLL_COS_PRE_EMBS_DEC_FIX_Lin_200_RELU_Lin_neubig_style_training/'
                       'SEED_models/lambda_20/seed_'+str(seed)+'/pred_5_test.txt')


list_lenghts=[]
list_ref=[]
for line in refs_file:
    line_split=line.split()
    list_lenghts.append(len(line_split))
    list_ref.append(line)


list_bas = []
for line in hyps_BASELINE_file:
    list_bas.append(line)

list_nll_cos = []
for line in hyps_NLL_COS_file:
    list_nll_cos.append(line)


average_value=sum(list_lenghts)/len(list_lenghts)

print (average_value)

print (len(list_ref))
print (len(list_bas))
print (len(list_nll_cos))


short_ref=open('SINGLE_BLEU/short_ref.txt','w')
short_bas=open('SINGLE_BLEU/short_bas.txt','w')
short_nll_cos=open('SINGLE_BLEU/short_nll_cos.txt','w')
long_ref=open('SINGLE_BLEU/long_ref.txt','w')
long_bas=open('SINGLE_BLEU/long_bas.txt','w')
long_nll_cos=open('SINGLE_BLEU/long_nll_cos.txt','w')


num_total=len(list_lenghts)
num_short=0
num_long=0
for i in range(len(list_lenghts)):
    if list_lenghts[i] < 20:
        num_short+=1
        short_ref.write(list_ref[i])
        short_bas.write(list_bas[i])
        short_nll_cos.write(list_nll_cos[i])
    else:
        num_long+=1
        long_ref.write(list_ref[i])
        long_bas.write(list_bas[i])
        long_nll_cos.write(list_nll_cos[i])

print (num_total)
print (num_short)
print (num_long)