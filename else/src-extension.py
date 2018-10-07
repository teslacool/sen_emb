import io
file1 = '../data/test.en'
file2 = '../data/test.de'

try:
    with io.open('outcome','w',encoding='utf8',errors='ignore',newline='\n') as tgt:
         f1 = io.open(file1,'r',encoding='utf8',errors='ignore',newline='\n')
         f2 = io.open(file2,'r',encoding='utf8',errors='ignore',newline='\n')
         for en,de in zip(f1,f2):
            if len(en.strip().split()) < 6 and len(de.strip().split()) < 6:
                print("en:%s      de: %s"%(en.strip(),de.strip()),file=tgt)
except Exception as e:
     print(str(e))

