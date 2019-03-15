

file_src_1=open('WMT17/ro-en/data/europarl-v8.ro-en.ro',encoding='utf-8')
file_tgt_1=open('WMT17/ro-en/data/europarl-v8.ro-en.en',encoding='utf-8')

file_src_2=open('WMT17/ro-en/data/SETIMES2.en-ro.ro',encoding='utf-8')
file_tgt_2=open('WMT17/ro-en/data/SETIMES2.en-ro.en',encoding='utf-8')

#file_src_3=open('WMT17/en-de/data/train/news-commentary-v13.de-en.de',encoding='utf-8')
#file_tgt_3=open('WMT17/en-de/data/train/news-commentary-v13.de-en.en',encoding='utf-8')


list_sentences_src_1=[]
list_sentences_tgt_1=[]
for line in file_src_1:
    line_sen=line.replace('\n','')
    line_sen=line_sen.strip()
    list_sentences_src_1.append(line_sen)

for line in file_tgt_1:
    line_sen=line.replace('\n','')
    line_sen = line_sen.strip()
    list_sentences_tgt_1.append(line_sen)


print (len(list_sentences_src_1))
print (len(list_sentences_tgt_1))

list_sentences_src_2=[]
list_sentences_tgt_2=[]
for line in file_src_2:
    line_sen=line.replace('\n','')
    line_sen=line_sen.strip()
    list_sentences_src_2.append(line_sen)

for line in file_tgt_2:
    line_sen=line.replace('\n','')
    line_sen = line_sen.strip()
    list_sentences_tgt_2.append(line_sen)

print (len(list_sentences_src_2))
print (len(list_sentences_tgt_2))

# list_sentences_src_3=[]
# list_sentences_tgt_3=[]
# for line in file_src_3:
#     line_sen=line.replace('\n','')
#     line_sen=line_sen.strip()
#     list_sentences_src_3.append(line_sen)
#
# print ('\n\nkka\n\n')
#
# for line in file_tgt_3:
#     line_sen=line.replace('\n','')
#     line_sen = line_sen.strip()
#     list_sentences_tgt_3.append(line_sen)
#
# print (len(list_sentences_src_3))
# print (len(list_sentences_tgt_3))


list_src_final=list_sentences_src_1+list_sentences_src_2#+list_sentences_src_3
list_tgt_final=list_sentences_tgt_1+list_sentences_tgt_2#+list_sentences_tgt_3

print (len(list_src_final))
print (len(list_tgt_final))

write_new_src=open('WMT17/ro-en/data/train.ro','w')
write_new_tgt=open('WMT17/ro-en/data/train.en','w')

for i in range(len(list_src_final)):
    write_new_src.write(list_src_final[i] + '\n')
    write_new_tgt.write(list_tgt_final[i] + '\n')
