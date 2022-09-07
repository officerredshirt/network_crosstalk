prefix = "tf-"
N = 2 	# number networks to generate

def get_ids():
	ids = []
	for i in range(N):
		ids.append(prefix + f"{i:06}")
	return ids

IDS = get_ids()
EXT = ["dir","bak","dat"]

DIR = "/mnt/c/Users/mindylp/Documents/python/network_crosstalk/"
RESDIR = DIR+"res/"

wildcard_constraints:
	id = prefix+"\d+"

rule all:
	input:
		expand(RESDIR+"{id}.xtalk",id=IDS)

rule gen_networks:
	output:
		expand(RESDIR+"{id}.arch.{ext}",id=IDS,ext=EXT)
	shell:
		DIR+f"src/get_networks.py -n {N} -p "+RESDIR+prefix+"; cp src/params.py "+RESDIR+prefix+"params.py"

rule get_achievables:
	input:
		RESDIR+"{id}.arch.bak", RESDIR+"{id}.arch.dir", RESDIR+"{id}.arch.dat"
	output:
		expand(RESDIR+"{id}.achieved.{ext}",ext=EXT,allow_missing=True)
	shell:
		DIR+"src/get_achievable_patterns.py -i "+RESDIR+"{wildcards.id}"


rule get_crosstalk:
	input:
		RESDIR+"{id}.achieved.bak", RESDIR+"{id}.achieved.dir", RESDIR+"{id}.achieved.dat"
	output:
		RESDIR+"{id}.xtalk"
	shell:
		DIR+"src/calc_crosstalk.py -i "+RESDIR+"{wildcards.id} -n 1"
