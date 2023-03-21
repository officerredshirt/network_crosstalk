import datetime

N = 2 	# number networks to generate

def get_ids():
	ids = []
	for i in range(N):
		ids.append(f"{i:06}")
	return ids

IDS = get_ids()

DIR = "/mnt/c/Users/mindylp/Documents/python/network_crosstalk/"
RESDIR = DIR+"test_res/"

DATABASE_PATH = RESDIR + "local_db.db"

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
CLUSTER_DIR = "cluster_" + TIMESTAMP

wildcard_constraints:
	id = "\d+"

rule all:
	input:
		expand(RESDIR+"{id}.xtalk",id=IDS)
	shell:
		"mkdir "+CLUSTER_DIR+"; cp -r Snakefile "+DIR+"src sm.sh " +CLUSTER_DIR+"||:; mv -b "+RESDIR+" "+CLUSTER_DIR

rule gen_networks:
	output:
		expand(RESDIR+"{id}.arch",id=IDS)
	shell:
		DIR+f"src/get_networks.py -n {N} -p "+RESDIR+" -d "+DATABASE_PATH 


rule get_achievables:
	input:
		RESDIR+"{id}.arch"
	output:
		expand(RESDIR+"{id}.achieved",allow_missing=True)
	shell:
		DIR+"src/get_achievable_patterns.py -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH


rule get_crosstalk:
	input:
		RESDIR+"{id}.achieved"
	output:
		RESDIR+"{id}.xtalk"
	shell:
		DIR+"src/calc_crosstalk.py -i "+RESDIR+"{wildcards.id} -n 3 -d "+DATABASE_PATH
