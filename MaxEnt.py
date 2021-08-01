'''
MAXENT - VIŠEKLASNA KLASIFIKACIJA
KLASIFIKACIJA GOVORA AMERIČKIH POLITIČARA

Korpus: govori američkih predsjednika Donalda Trumpa i
Barack Obame te senatorice Hillary Clinton

Napraviti sustav za prepoznavanje govora tj. određivanje osobina autora
prema odabranom tekstu 

Za to nam je potrebna višeklasna logistička regresija koja se još zove
– Softmax regresija
– Maxent klasifikator

Sustav za prepoznavanje govora se može napraviti uz pomoć logističke regresije tj. MaxEnt modela.
Definiraju se osobine nad mjestima u podacima gdje osobine predstavljaju skupove mjesta u podacima
koji su dovoljno karakteristični da zasluže parametre modela: riječi, riječi s brojem, riječi koje
završavaju na "iti", itd.

Osobine se često dodavaju tijekom razvoja modela kako bi označili pogreške. Često i
najjednostavnije je dodavati osobine koje označavaju loše kombinacije.

Cilj logističke regresije je učenje klasifikatora koji može donijeti binarnu odluku za
klasu nekog novog ulaznog promatranja. 

Komponente logističke regresije:
Strojno učenje za klasifikaciju ima sljedeće komponente:
1. Reprezentacija osobina za ulaze (svaki ulaz je predstavljen vektorom osobina)
2. Funkcija klasifikacije koja računa procjenu klase (softmax)
3. Aktivacijska funkcija za učenje koja obično uključuje minimizaciju greške (unakrsna entropija gubitka)
4. Algoritam za optimizaciju aktivacijske funkcije (stohastičko opadanje gradijenta)

Faze logističke regresije:
1. Učenje (treniranje) pomoću stohastičkog opadanja gradijenta i gubitka unakrsne entropije
2. Testiranje (računa se i vraća klasa s većom vjerojatnošću)

Osim logističke regresije u radu će se raditi deskriptivna statistika. Deskriptivna statistika se
bavi mjerama centralne tendencije (aritmetička sredina, medijan i mod),
mjerama varijabiliteta (raspon, standardna devijacija, varijanca, interkvartilni raspon, semiinterkvartilni
raspon i prosječno odstupanje), kao i grafičkim i tabelarnim prikazivanjem osnovnih statističkih vrijednosti.
U statističkom žargonu, deskriptivna statistika se naziva statistikom sa malim s jer je
osnovni cilj deskriptivne statistike da ponudi podatke koji se dalje mogu obrađivati.

Koraci u sustavu prepozavanja govora:
1. Mjere centralne tendencije (aritmetička sredina, medijan i mod,raspon)
2. Mjere varijabiliteta (standardna devijacija, varijanca i standardna pogreška)
3. Grafički prikazati osnovne statističke vrijednosti na histogramu
4. Značajke za svaki govor:
    4.1. Izračunati [log (duljina (d))] gdje je d govor određenog autora
    4.2. Izračunati [brojRijeci(pozitivne) ∈ d] gdje pozitivne predstavljaju broj pozitivnih riječi 
    4.3. Izračunati [brojRijeci(negativne) ∈ d] gdje negativne predstavljaju broj negativnih riječi 
    4.4. Izračunati [brojRijeci("Terorizam") ∈ d] gdje treba pronaći koliko ima riječi "terorizam" za svakog govornika 
    4.5. Izračunati [brojRijeci("sr") ∈ d] gdje sr predstavlja broj specifičnih riječi u listi "specifične riječi"
    4.6. Izračunati [brojRijeci("America") ∈ d] gdje treba pronaći riječ "America" za svakog govornika
10. Napraviti funkciju klasifikacije softmax za svaku klasu
11. Napraviti troškovnu funkciju 
12. Napraviti metodu opadanja gradijenta - optimizacija funkcije gubitka
'''

#dodavanje modula
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import re
from nltk.tokenize import regexp_tokenize
from random import shuffle,uniform
from math import e , pow , log

#tokenizacija
def tokeniziraj(tekst):
    uzorak = r'(?:[A-Za-z]{1,3}\.)+|\w+(?:[-\'_]\w+)*|\$?\d+(?:\.\d+)*%?'
    return regexp_tokenize(tekst, uzorak)

#lista zaustavnih riječi za engleski jezik
def stopWords():
    tekst = open("english.stop","r",encoding = "utf8").read()
    return tekst.split("\n")

stopwords = set(stopWords()) 

def readFile(filename):
    return open(filename,"r").read()

#pozitivne i negativne riječi 
pozitivne = readFile("pozitivne.txt").split("\n")
negativne = readFile("negativne.txt").split("\n")

#klase 
klase = ["Trump","Obama","Clinton"]

#specifične riječi
specificneRijeci = ["war","terrorism","money","politics","politician","president","democrat","republican","healthcare",
                 "Iraq","stupid","tremendous","winning","amazing","great","military","classy","security"]
def scalar(x,y):
    return sum(a*b for a,b in zip(x,y))

'''Softmax funkcija
Softmax funkcija se koristi za izračunavanje vjerojatnosti tako da
ulaz x pripada svakoj klasi. Za ulazni vektor z = (z1, ..., zn)
softmax funkcija je definirana sa sljedećom formulom:'''
#softmax(z)=(ez1∑nk=1ezk,...,ezn∑nk=1ezk) kopirati formulu sa prezentacije 

def softmax(z):
    l = len(z)
    a = sum(pow(e,x) for x in z)
    probs = [0]*l 
    for i in range(0,l):
        probs[i] = pow(e,z[i]) / a
    return probs

'''
Korpus - podaci za treniranje i testiranje
Tri datoteke u kojoj folder naziva Trump sadrži 2577 izjava, u folderu Obama
ima 2007 govora, a u folderu Clinton ima 3062 govora.
U prvom slučaju 80% svih govora koristit će se za trening, a ostatak za
testiranje. 
'''

#dataset svaka lista je lista riječi za jednu izjavu 
trump = []
obama = []
clinton = []

#Prima liniju i razbija je na rijeci
def select_data(niz):
    lista=[]
    for el in niz:
        for temp in el:
            if(temp != ""):
                lista.append(temp)  
    return lista


def import_data(equal = False):
    trump = []
    obama = []
    clinton = []

    #for trump
    for i in range(0,100): #2576
        trump.append(list(set(tokeniziraj(readFile("Trump\DonaldTrump("+str(i)+").txt")))-stopwords)) #dodajemo listu riječi za jednu izjavu 
    #for obama
    for i in range(0,100): #2006
        obama.append(list(set(tokeniziraj(readFile("Obama\BarackObama("+str(i)+").txt")))-stopwords))
    #for clinton
    for i in range(0,100): #3061
        clinton.append(list(set(tokeniziraj(readFile("Clinton\HillaryClinton("+str(i)+").txt")))-stopwords))
    shuffle(trump)
    shuffle(obama)
    shuffle(clinton)
    
    return trump,obama,clinton
trump,obama,clinton=import_data()
arrayTrump=select_data(trump)
arrayObama=select_data(obama)
arrayClinton=select_data(clinton)


'''
Značajke za svakog autora
Za svaki skup podataka jedan je od prvih koraka normalizacija kako bi se
iz govora izrezale riječi zaustavljanja radi lakšeg računanja. Nakon toga
je lako dobiti broj pozitivnih/negativnih riječi s posebnog popisa.
Za ovaj problem značajke su karakteristike lingvistike za tekst. Značajka je
numerički vektor koji je definiran u nastavku:
značajka = (log(len(govor)), broj_pozitivnih_rijeci, broj_negativnih_rijeci,
broj_specijalnih_rijeci, broj_terorizam_rijeci, broj_amrica_rijeci).
Ispod je funkcija za dobivanje značajki za svaki govor.

'''
def features(text):#  tekst je lista rijeci od jedne izjave 
    f = [0]*6 # vektor značajki 
    f[0] = log(len(text),10) #prva značajka je log (length(speech)) sa bazom 10 
    pos,neg,special,america,terror = 0,0,0,0,0
    for word in text:
        if word in pozitivne:
            pos += 1
        elif word in negativne:
            neg += 1
        if word in specificneRijeci:
            special += 1
        w = word.lower()
        if w in ["american","americans","america","usa","u.s.a","us"]:
            america += 1
        elif w in ["terrorism","terrorist"]:
            terror += 1
    f[1] = pos
    f[2] = neg
    f[3] = special
    f[4] = america
    f[5] = terror
    return f
'''
Učenje
Glavni cilj ovog sustava je dobra odluka o novim nevidljivim uzorcima.
To se rješava pronalaženjem vektora težine w i pristranosti b za svaku
od 3 klase. Težine predstavljaju posebnosti svake pojedine osobine.
'''

def prepare_data(equal = False):
    train = [] #train-set
    test = [] #test-set

    t = len(trump)
    o = len(obama)
    z = len(clinton)

    #postatak 80% za treniranje
    a=round(0.8*t) 
    b=round(0.8*o) 
    c=round(0.8*z) 
    #print(a,b,c)
    
    for i in range(0,t): #for trump
        if i <= a:
            train.append((features(trump[i]),"Trump"))
        else:
            test.append((features(trump[i]),"Trump"))
    for i in range(0,o): #for obama
        if i <= b:
            train.append((features(obama[i]),"Obama"))
        else:
            test.append((features(obama[i]),"Obama"))    
    for i in range(0,z): #for clinton
        if i <= c:
            train.append((features(clinton[i]),"Clinton"))
        else:
            test.append((features(clinton[i]),"Clinton"))
    shuffle(train)
    shuffle(test)
    return train,test


train_set , test_set = prepare_data() # dobavimo podatke

#Troškovna funkcija
'''Sljedeće što je potrebno izračunati je troškovna funkcija
tj. potrebno je izmjeriti koliko je naša procjena dobra za
klase.
Ovakva metrika ili udaljenost se naziva troškovnom funkcijom.

Funkcija troška za jedan primjer x = (x1, ..., xn) i procijenjene
klase y je zbroj zapisa K klasa:
LCE (y ^, y) = - ∑nk = 11 {y = k} logp (y = k | x) gdje je
{y = k} 1 ako je y= k, inače je 0. Cilj je pronaći takve vektore težine
W1, W2, W3 za 3 klase, tako da je funkcija troška minimizirana.
Gradijent za jednostvani uzorak x = (x1, ..., xn) je
∂LCE/∂wk = (1 {y = c} −p (y = c | x)) xk.
Postoji razlika između prave klase c i vjerojatnosti da model iznosi c,
pomnožen s xk za svaki k = 1,...,n

Vektori
Vektori težine su slučajni brojevi za početak tako da učenje
nije determinističko.'''

w=[]
learn_rate = 0.10 #postavimo beta razinu za učenje
epochs = 8
def setWeights():  
    w_trump = [uniform(-0.5,0.5) for i in range(7)]
    w_obama = [uniform(-0.5,0.5) for i in range(7)]
    w_clinton = [uniform(-0.5,0.5) for i in range(7)]
    w = [w_trump,w_obama,w_clinton]  #
    print("#####")
    print(w_trump)
    return w
w = setWeights()


#Učenje

'''Skup podataka pretvoren u skup značajki
Inicijaliziramo vektore težina za svaku klasu, postavite mjeru učenja β
i broj epoha za svaku epohu.

Za svaki uzorak iz treninga pronađemo vjerovatnost da uzorak pripada
određenoj klasi.

Kako postoje 3 klase c1, c2, c3, onda je broj epohe 1, a stopa učenja 0,1.
Za uzorak x je dobiven iz x = (1,1,2) i njegova prava klasa je c = c1.
Vektori za klase su w1 = (0,1,2,1,0), w2 = (0,5,1,1,1), w3 = (1,0,0,1,0.3).

Prvo što treba učiniti je izračunati vjerojatnosti za svaku klasu,
da se dobije uzorak x koji pripada određenoj klasi.
Za svaku klasu c izračunava se vektor vjerovatnosti uz pomoć softmax
funkcije i iznosi (0,389,0.582,0.029). To znači da je najvjerojatnija klasa c2.
Sljedeći korak je izračunavanje novih težina, što se vrši računanjem
vektora parcijalnih derivacija, ovdje ćemo pokazati samo za jedan razred, c1.
Najvažniji dio je razlika između prave klase c1 i predviđene klase c2.
Budući da su te dvije klase različite, razlika između njih je
0-prob (c1) = - 0,389.
Vektor djelomičnih izvedenica je sada (−0.389, -0.389, -0.778, -0.389),
posljednji broj je djelomična izvedba troškovne funkcije za varijabilnu
pristranost.
Posljednje što treba učiniti je ažurirati vektore za klasu c1.'''

def train():
    w = setWeights()
    for epoch in range(epochs):
        shuffle(train_set)
        for sample in train_set:
            f = sample[0] 
            z = [0,0,0] 
            for j in range(0,3):
                z[j] = scalar(w[j][:6],f) + w[j][6]
                if z[j] > 600: 
                    z[j] = 600
            prob = softmax(z) 
            c = sample[1] 
         
            for j in range(0,3):
                dif = (1 if c == klase[j] else 0) - prob[j] 
                grad = [dif * x for x in f] 
                grad.append(dif) 
                w[j] = [x - learn_rate * y for x,y in zip(w[j],grad)] 
    return w


'''Za učenje se koristi algoritam stohastičkog gradijenta opadanja.
To je numerička metoda koja se koristi u slučajevima kada je pokušaj
pronalaska najmanje funkcije analitički teži od pronalaska iterativne metode.
Prikladno je koristiti ovaj algoritam za konveksne funkcije jer takve
vrste funkcija imaju samo 1 minimum (globalni minimum) i algoritam se ne
može zaglaviti u lokalnim minimumima.'''

'''
Evaluacija (vrednovanje)
Parametri evaluacije modela koji se koriste su: preciznost, opoziv i f1 mjera.
tp - Istina pozitivan #točno ili istina triba viditi
fp - Lažno pozitivan
tn - Istina negativan
fn - Lažno negativan

Za svaku od 3 klase kreiraju se tablice za slučajnosti, a zatim se
izrađuje zajednička tablica.
'''
def classify(weights,features):
    z = [0,0,0] 
    for j in range(0,3):
        z[j] = scalar(weights[j][:6],features) + weights[j][6]
        if z[j] > 600: 
            z[j] = 600
    prob = softmax(z)
    return klase[prob.index(max(prob))] 



ctables = [[0,0,0,0],[0,0,0,0],[0,0,0,0]] 
def evaluate(weights):
    for (X,c) in test_set: 
        y = classify(weights,X) 
        for j in range(0,3):
            if klase[j] == y :
                if y == c: 
                    ctables[j][0] +=1 
                else:
                    ctables[j][1] +=1 
            else:
                if y == c:
                    ctables[j][3] += 1 
                else: 
                    ctables[j][2] += 1 

evaluate(train())
#izračunavanje preciznosti, opoziva i f1 mjere 
tp = ctables[0][0]+ctables[1][0]+ctables[2][0]
fp = ctables[0][1]+ctables[1][1]+ctables[2][1]
tn = ctables[0][2]+ctables[1][2]+ctables[2][2]
fn = ctables[0][3]+ctables[1][3]+ctables[2][3]

 
p = tp/(tp+fp) # preciznost 
r = tp/(tp+tn) # opoziv
f = (2*p*r)/(p+r) # f1 mjera
print("PRECIZNOST: ",p)
print("OPOZIV: ",r)
print("F1 MJERA: ",f)


#STATISTIKA
# Mjere centralne tendencije (aritmetička sredina, medijan, mod i raspon)
# Mjere varijabiliteta (standardna devijacija, varijanca i standardna pogreška)
# Grafički prikazati osnovne statističke vrijednosti na histogramu

#Kreirajmo skup podataka s kojim ćemo raditi i crtati histogram za vizualizaciju:

matplotlib.style.use('ggplot')
'''

data=arrayTrump
plt.hist(data, bins=len(arrayTrump), range=(0,len(arrayTrump)), edgecolor='black')
plt.xlabel("Rijec")
plt.ylabel("Frekvencija")
plt.title("Trump")
plt.show()



data=arrayObama
plt.hist(data, bins=len(arrayObama), range=(0,len(arrayObama)), edgecolor='black')
plt.xlabel("Rijec")
plt.ylabel("Frekvencija")
plt.title("Obama")
plt.show()

data=arrayClinton
plt.hist(data, bins=len(arrayClinton), range=(0,len(arrayClinton)), edgecolor='black')
plt.xlabel("Rijec")
plt.ylabel("Frekvencija")
plt.title("Clinton")
plt.show()
'''
'''
MJERE SREDIŠNJE TENDENCIJE
Mjere središnje tendencije uključuju aritmetički sredinu, medijan i mod.
Aritmetička sredina se izračunava po formuli:
μ = ΣNi xi/N
Medijan je srednja vrijednost koja se uzima po formuli
n+1/2 za sortirane podatke.
Mod je najčešća vrijednost.
'''
rijecnikTrump={}
for rijec in arrayTrump:
    if(rijec in rijecnikTrump):
        rijecnikTrump[rijec] = rijecnikTrump.get(rijec, 0) + 1
    else:
        rijecnikTrump[rijec]=1
#print(rijecnikTrump)
#print("*******************************")
rijecnikObama={}
for rijec in arrayObama:
    if(rijec in rijecnikObama):
        rijecnikObama[rijec] = rijecnikObama.get(rijec, 0) + 1
    else:
        rijecnikObama[rijec]=1
#print(rijecnikObama)
#print("*******************************")
rijecnikClinton={}
for rijec in arrayClinton:
    if(rijec in rijecnikClinton):
        rijecnikClinton[rijec] = rijecnikClinton.get(rijec, 0) + 1
    else:
        rijecnikClinton[rijec]=1
#print(rijecnikClinton)
#print("*******************************")

'''ARITMETIČKA SREDINA
Modul Numpy sadrži funkciju mean za izračunavanje aritmetičke  sredine
'''
#mean = np.mean(list(rijecnikTrump.values()))
#mean
print("****************DESKRIPTIVNA STATISTIKA****************")
print("AR.SREDINA")
#print(mean)
print("TRUMP: ",np.mean(list(rijecnikTrump.values())))
print("OBAMA: ",np.mean(list(rijecnikObama.values())))
print("CLINTON: ",np.mean(list(rijecnikClinton.values())))

'''MEDIJAN
Modul Numpy sadrži median funkciju za izračun medijana.
'''
print("\nMEDIJAN")
print("TRUMP: ",np.median(list(rijecnikTrump.values())))
print("OBAMA: ",np.median(list(rijecnikObama.values())))
print("CLINTON: ",np.median(list(rijecnikClinton.values())))



'''MOD
Iz našeg histograma je vidljivo da imamo neke zapažene vrijednosti.
Ne postoji ugrađena funkcija iz numpy modula ali postoji u modulu
scipy stats koji možemo koristiti.
'''
print("\nMOD")

mode = stats.mode(list(rijecnikTrump.values()))
print("TRUMP: The modal value is {} with a count of {}".format(mode.mode[0], mode.count[0]))

mode = stats.mode(list(rijecnikObama.values()))
print("OBAMA: The modal value is {} with a count of {}".format(mode.mode[0], mode.count[0]))

mode = stats.mode(list(rijecnikClinton.values()))
print("CLINTON: The modal value is {} with a count of {}".format(mode.mode[0], mode.count[0]))


#print(mode)

'''
RASPON VRIJEDNOSTI
Raspon daje mjeru za raspodjelu vrijednosti.
Raspon se jednostavno izračunava kao maksimalna
vrijednost - minimalna vrijednost
Max (xi) - min (xi)
Numpy sadrži funkciju np.ptp.'''
print("\nRASPON VRIJEDNOSTI")
print("TRUMP: ",np.ptp(list(rijecnikTrump.values())))
print("OBAMA: ",np.ptp(list(rijecnikObama.values())))
print("CLINTON: ",np.ptp(list(rijecnikClinton.values())))

'''
VARIJANCA
Varijanca je mjera za varijabilnost podataka, izračunava se kao:
σ2 = ΣNi(xi-μ)^2/N
Numpy sadrži varijancu kao funkciju np.var()'''
print("\nVARIJANCA")

print("TRUMP: ",np.var(list(rijecnikTrump.values())))
print("OBAMA: ",np.var(list(rijecnikObama.values())))
print("CLINTON: ",np.var(list(rijecnikClinton.values())))

'''
STANDARDNA DEVIJACIJA 
Varijanca može biti ogromna za velike skupove podataka pa ćemo koristiti
standardnu devijaciju, a to je kvadratni korijen varijance:
σ = sqrt(σ^2)
Numpy sadrži funkciju np.std()
'''
print("\nSTANDARDNA DEVIJACIJA")
print("TRUMP: ",np.std(list(rijecnikTrump.values())))
print("OBAMA: ",np.std(list(rijecnikObama.values())))
print("CLINTON: ",np.std(list(rijecnikClinton.values())))

'''
STANDARDNA POGREŠKA ILI ODSTUPANJE
Standardna pogreška od sredine procjenjuje varijabilnost između
uzorka sredine ako bismo uzeli više uzoraka iz iste populacije.
Standardna pogreška sredine računa promjenu između uzoraka
za razliku od standardne devijacije koja mjeri
promjenu unutar jednog uzorka.
Izračunava se po formuli:
SE = s/sqrt(n)
gdje je s standardna devijacija uzorka.
Numpy ne sadrži funkciju (iako ju je lako izračunati), pa umjesto
toga se može upotrijebiti modul scipy stats'''

print("\nSTANDARDNA POGREŠKA ILI ODSTUPANJE")

print("TRUMP: ",stats.sem(list(rijecnikTrump.values())))
print("OBAMA: ",stats.sem(list(rijecnikObama.values())))
print("CLINTON: ",stats.sem(list(rijecnikClinton.values())))

'''
Zaključak projekta
Svrha ovog projekta je primjena NLP-a i strojnog učenja naučenih kroz
obradu prirodnog jezika. Parametri evaluacije nisu zadovoljavajući jer
se u ovom radu koriste vrlo jednostavne značajke jezika. Za dobivanje
puno boljih evaluacijskih parametara potrebno je uključiti neke
karakterstike specifične za jezik koje imaju veći utjecaj na statističko
učenje za razliku od ovih značajki. Isto tako treba razumjeti i implementirati
jedan algoritam optimizacije, u ovom slučaju je to stohastički gradijent
opadanja i objasniti njegovu ulogu u učenju tj. pronaći najbolje težine
za svaku klasu.
'''
