import xml.etree.ElementTree as ET


tree_src=ET.parse('IWSLT_2018/Basque_English/test_data/eu-en/IWSLT18.TED.tst2018.eu-en.eu.xml')

root_src=tree_src.getroot()

child=root_src[0]

list_sentences_src=[]

for doc in child:
    for sen in doc.findall('seg'):
        text_sen=sen.text
        text_sen=text_sen.replace('\n','')
        text_sen = text_sen.strip()
        list_sentences_src.append(text_sen)


#tree_tgt = ET.parse('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/IWSLT18.TED.dev2018.eu-en.en.xml')

#root_tgt = tree_tgt.getroot()

#child = root_tgt[0]

#list_sentences_tgt = []

#for doc in child:
#    for sen in doc.findall('seg'):
#        text_sen = sen.text
#        text_sen = text_sen.replace('\n', '')
#        text_sen = text_sen.strip()
#        list_sentences_tgt.append(text_sen)


print (len(list_sentences_src))
#print (len(list_sentences_tgt))

write_file_src=open('IWSLT_2018/Basque_English/test_data/eu-en/test.eu','w',encoding='utf-8')
#write_file_tgt=open('IWSLT_2018/Basque_English/In_domain/train_dev/eu-en/dev.en','w',encoding='utf-8')

for i in range(len(list_sentences_src)):
    write_file_src.write(list_sentences_src[i]+'\n')
#    write_file_tgt.write(list_sentences_tgt[i]+'\n')
