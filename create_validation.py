file_2008en=open('WMT17_news_data/dev_test_data/newstest2008.en')
file_2008de=open('WMT17_news_data/dev_test_data/newstest2008.de')
file_2009en=open('WMT17_news_data/dev_test_data/newstest2009.en')
file_2009de=open('WMT17_news_data/dev_test_data/newstest2009.de')
file_2010en=open('WMT17_news_data/dev_test_data/newstest2010.en')
file_2010de=open('WMT17_news_data/dev_test_data/newstest2010.de')
file_2011en=open('WMT17_news_data/dev_test_data/newstest2011.en')
file_2011de=open('WMT17_news_data/dev_test_data/newstest2011.de')
file_2012en=open('WMT17_news_data/dev_test_data/newstest2012.en')
file_2012de=open('WMT17_news_data/dev_test_data/newstest2012.de')
file_2013en=open('WMT17_news_data/dev_test_data/newstest2013.en')
file_2013de=open('WMT17_news_data/dev_test_data/newstest2013.de')


write_val_en=open('WMT17_news_data/dev_test_data/val.newstest.08-13.en','w')
write_val_de=open('WMT17_news_data/dev_test_data/val.newstest.08-13.de','w')

list_2008_en=[]
list_2008_de=[]
list_2009_en=[]
list_2009_de=[]
list_2010_en=[]
list_2010_de=[]
list_2011_en=[]
list_2011_de=[]
list_2012_en=[]
list_2012_de=[]
list_2013_en=[]
list_2013_de=[]


for line in file_2008en:
    write_val_en.write(line)
for line in file_2009en:
    write_val_en.write(line)
for line in file_2010en:
    write_val_en.write(line)
for line in file_2011en:
    write_val_en.write(line)
for line in file_2012en:
    write_val_en.write(line)
for line in file_2013en:
    write_val_en.write(line)



for line in file_2008de:
    write_val_de.write(line)
for line in file_2009de:
    write_val_de.write(line)
for line in file_2010de:
    write_val_de.write(line)
for line in file_2011de:
    write_val_de.write(line)
for line in file_2012de:
    write_val_de.write(line)
for line in file_2013de:
    write_val_de.write(line)