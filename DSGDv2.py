__author__ = 'bread'
from pyspark import SparkContext, SparkConf
from numpy import random, dot, zeros
from copy import deepcopy
import math
import sys
#from random import shuffle


def GetVal(s):
    sv = s.split(',')
    tmp = []
    tmp0 = int(sv[0])-1   ##  W
    tmp1 = int(sv[1])-1   ##  H
    ind = work*int(tmp0/Wn) + int(tmp1/Hn)
    for sth in sv:
        tmp.append(int(sth))
    return (ind, tmp)


def calc(s):
    sv = s.split(',')
    tmp0 = int(sv[0])   ##  W
    tmp1 = int(sv[1])   ##  H
    Ni = [0 for i in range(WnAll)]
    Nj = [0 for i in range(WnAll)]
    Ni[tmp0-1] = 1
    Nj[tmp1-1] = 1
    return [Ni, Nj, 1]


def SGDfirst(x):
    fa = deepcopy(Fac.value)
    Wtm = dict()
    Htm = dict()
    lambv = deepcopy(lambb.value)
    Ni = deepcopy(NI.value)
    Nj = deepcopy(NJ.value)
    Nprime = int(deepcopy(Npri.value))
    for key, val in x:
        for ij in val:
            ######################
            Nprime += 1
            epsi = (100 + Nprime) ** (-float(Beta.value))
            tmp = deepcopy(ij[2])
            W = deepcopy(Wb.value.get(ij[0]-1))
            H = deepcopy(Hb.value.get(ij[1]-1))
            for i in range(fa):
                tmp -= (W[i]*H[i])
                #if(math.isnan(tmp)):
                    #print Wb.value.get(ij[0]-1), Wtm.get(ij[0]-1), '\n\n\n'
            gradW = (-2)*tmp*H + 2*lambv/Ni[ij[0]-1]*W
            Wtm[ij[0]-1] = deepcopy(W - epsi*gradW)
            gradH = tmp*(-2)*W + 2*lambv/Nj[ij[1]-1]*H
            Htm[ij[1]-1] = deepcopy(H - epsi*gradH)
            ######################
    #Htm.cache()
    return ([{0:Wtm, 1:Htm}])



def Red(x, y):
    return {0: dict(x.get(0), **y.get(0)), 1: dict(x.get(1), **y.get(0))}


WnAll = 500
HnAll = 500

fac = int(sys.argv[1])
work = int(sys.argv[2])
iter = int(sys.argv[3])
beta = float(sys.argv[4])
lamb = float(sys.argv[5])
Vpath = sys.argv[6]
Wpath = sys.argv[7]
Hpath = sys.argv[8]
Wn = WnAll/work  # num of elements per row per block
Hn = HnAll/work  # num of elements per row per block


conf = SparkConf().setAppName("sb605").setMaster("local").set("spark.cores.max", work)
sc = SparkContext(conf=conf)


W = dict() ## change to array if possible
H = dict()
for i in range(WnAll): ## init W and H
    W[i] = random.ranf(size=fac)
for i in range(HnAll):
    H[i] = random.ranf(size=fac)

Wb = sc.broadcast(W)
Hb = sc.broadcast(H)
lambb = sc.broadcast(lamb)
Fac = sc.broadcast(fac)
Beta = sc.broadcast(beta)

lines = sc.textFile(Vpath)
V = lines.map(lambda x: (GetVal(x)))
V.cache()
Set = lines.map(lambda x: (calc(x))).reduce(lambda a, b: [map(sum, zip(a[0], b[0])), map(sum, zip(a[1], b[1])), a[2]+b[2]])
NI = sc.broadcast(Set[0])   # static
NJ = sc.broadcast(Set[1])  # static
N0 = Set[2]
Npri = sc.broadcast(0)



seq = [[0 for x in range(work)] for x in range(work)]
for i in range(work):
    for j in range(i, work):
        seq[i][j] = i+(j-i)*(work+1)
    for j in range(0, i):
        seq[i][j] = (j+work-i)*work+j


####################################################### SGD
first = 0
res = dict()
for i in range(iter):
    for j in range(work):
        res = sc.parallelize(V.filter(lambda (x, y): x in seq[j]).groupByKey().collect(), work).mapPartitions(SGDfirst).reduce(lambda x, y: Red(x, y))
        res = {0: dict(W, **res.get(0)), 1: dict(H, **res.get(1))}
        W = deepcopy(res.get(0))
        H = deepcopy(res.get(1))
        Wb.unpersist()
        Wb = sc.broadcast(H)
        Hb.unpersist()
        Hb = sc.broadcast(H)
        nn = Npri.value
        Npri.unpersist()
        Npri = sc.broadcast(float(i+j/work)*N0)

########################################################### reconstruction error
Wm = zeros((500, fac))
Hm = zeros((fac, 500))
Vm = zeros((500, 500))


with open(Vpath, "r") as ins:
    for line in ins:
        sv = line.split(',')
        Vm[int(sv[0])-1][int(sv[1])-1] = int(sv[2])

tmpW = res.get(0)
for key in range(500):
    tmprow = tmpW.get(key)
    for val in range(fac):
        Wm[key][val] = tmprow[val]

tmpW = res.get(1)
for key in range(500):
    tmprow = tmpW.get(key)
    for val in range(fac):
        Hm[val][key] = tmprow[val]



sth = dot(Wm, Hm)
print Wm, Hm, Vm
loss = 0
mode = 0
for key in range(500):
    for ke2 in range(500):
        if Vm[key][ke2] != 0:
            loss += (Vm[key][ke2] - sth[key][ke2])**2
            mode += Vm[key][ke2] ** 2
print loss, mode, loss/mode

####################################################################output

f = open(Wpath, 'w')
tmpW = res.get(0)
for key in range(WnAll):
    tmprow = tmpW.get(key)
    f.write(str(tmprow[0]))
    for val in range(fac-1):
        f.write(','+str(tmprow[val+1]))
    f.write('\n')
f.close()


f = open(Hpath, 'w')
tmpW = res.get(1)
for key in range(WnAll):
    tmprow = tmpW.get(key)
    f.write(str(tmprow[0]))
    for val in range(fac-1):
        f.write(','+str(tmprow[val+1]))
    f.write('\n')
f.close()

#shuffle(seq)numpy.savetxt("foo.csv", a, delimiter=",")
