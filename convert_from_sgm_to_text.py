sgm_file=open('WMT17/ro-en/data/newstest2016-roen-ref.en.sgm')

text_file=open('WMT17/ro-en/data/test.en','w')

for line in sgm_file:
    if "<seg id=" in line:
        new_line=line.replace('</seg>','')
        new_line = new_line.replace('\n', '')
        new_line=new_line.split('">')
        if len(new_line)>2:
            sentence='">'.join(new_line[1:])
        else:
            sentence=new_line[1]

        text_file.write(sentence+'\n')