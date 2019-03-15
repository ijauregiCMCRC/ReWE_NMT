

file_src=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.tags.eu-en.eu',encoding='utf-8')
file_tgt=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.tags.eu-en.en',encoding='utf-8')


list_sentences_src=[]
list_sentences_tgt=[]
for line in file_src:
    if line.startswith('<')!=True:
        line_sen=line.replace('\n','')
        line_sen=line_sen.strip()
        list_sentences_src.append(line_sen)

for line in file_tgt:
    if line.startswith('<')!=True:
        line_sen=line.replace('\n','')
        line_sen = line_sen.strip()
        list_sentences_tgt.append(line_sen)


print (len(list_sentences_src))
print (len(list_sentences_tgt))

write_new_src=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.eu','w')
write_new_tgt=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/train.en','w')

for i in range(len(list_sentences_src)):
    write_new_src.write(list_sentences_src[i]+'\n')
    write_new_tgt.write(list_sentences_tgt[i] + '\n')
