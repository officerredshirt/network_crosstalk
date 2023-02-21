import dill

"""
filename = "test-000000.xtalk"
optres = dill.load(open("./test_res/"+filename, "rb"))
print(optres)
"""

filename = "cluster_2022-11-08/res/kpr-000001.xtalk"
optres = dill.load(open(filename,"rb"))
print(optres)
