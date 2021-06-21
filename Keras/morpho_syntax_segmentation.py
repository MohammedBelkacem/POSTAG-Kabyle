uḍfiren=[]
uzwiren=[]
Asenqeḍ=['...',',',';','?','!',':','"','(',')','*','_','.','[',']','{','}','«','»','+','=','“','”']
#ad d-nekkes akk imqimen, tizelɣiwin ... i yettentaḍen d yimyagen, ismawen, tinzaɣ.....atg seg ufaylu imqimen_uzwiren_uḍfiren.txt
for i in open("imqimen_uzwiren_uḍfiren.txt ",encoding='utf-8'):    a=i.replace("\ufeff","").replace("\n","").strip()
    if (a[len(a)-1]=="-"):
     uzwiren.append(str(a).lower())
    else:
       uḍfiren.append(str(a).lower())

# tawuri-a tbeṭṭu awal
def bḍu_awal(awal,uḍfiren,uzwiren):
    amurfim=awal[0:awal.find('-')+1]
    awal_yebḍan=''
    if (amurfim in uzwiren):
        awal=awal[awal.find('-')+1:len(awal)]
        awal_yebḍan=awal_yebḍan+' '+amurfim
        while awal.find('-')>=0:
            amurfim=awal[0:awal.find('-')+1]
            awal=awal[awal.find('-')+1:len(awal)]
            awal_yebḍan=awal_yebḍan+' '+amurfim
        awal_yebḍan=awal_yebḍan+' '+awal

    else:
        amurfim=awal[0:awal.find('-')]
        awal_yebḍan=awal_yebḍan+' '+amurfim
        awal=awal[awal.find('-')+1:len(awal)]
        while awal.find('-')>=0:
            amurfim=awal[0:awal.find('-')]
            awal_yebḍan=awal_yebḍan+' '+'-'+amurfim
            awal=awal[awal.find('-')+1:len(awal)]
    if ('-'+awal in uḍfiren):
         awal_yebḍan=awal_yebḍan+' '+'-'+awal
    else:
         awal_yebḍan=awal_yebḍan
    return awal_yebḍan
# tawuri-a tbeṭṭu tafyirt
def bḍu_tafyirt(tafyirt,uḍfiren,uzwiren):
       a= tafyirt.split()
       tafyirt1=""
       for i in a: #awalen
        if(i.find('-')<0):
            tafyirt1=tafyirt1+' '+i
        else:
                                                                                
       awals=bḍu_awal(i,uḍfiren,uzwiren)
       tafyirt1=tafyirt1+' '+awals
      tafyirt1=tafyirt1.strip()
      return tafyirt1

f= open("tokenized_text.txt","w+", encoding='utf-8') 
g=open("brut_text.txt",encoding='utf-8')
for adur in g:
    for i in Asenqeḍ:
        adur=adur.replace(i,' '+i+' ').replace("\ufeff","")
    izirig=bḍu_tafyirt(adur,uḍfiren,uzwiren)
    izirig=izirig.replace("  "," ")
    f.write(izirig+'\n')
f.close()
g.close() 

