import datetime

IDS = [f"{0:06}"] # NOTE this is hard-coded in later

DIR = "/mnt/c/Users/Melinda/Documents/python/network_crosstalk/"
RESDIR = DIR+"res/"

DATABASE_PATH = RESDIR + "local_db.db"

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
CLUSTER_DIR = "cluster_" + TIMESTAMP

wildcard_constraints:
	id = "\d+"

rule all:
	input:
		expand(RESDIR+"{id}.xtalk",id=IDS)
	shell:
		"mkdir "+CLUSTER_DIR+"; cp -r Snakefile "+DIR+"src sm.sh " +CLUSTER_DIR+ \
            "||:; mv -b "+RESDIR+" "+CLUSTER_DIR

rule gen_models:
    output:
        RESDIR+"tf_chrom_equiv_pr_bound.out", RESDIR+"tf_pr_bound.out", RESDIR+"kpr_pr_open.out"
    shell:
        DIR+f"src/gen_kinetic_models.py "+RESDIR

rule gen_networks:
	output:
		expand(RESDIR+"{id}.arch",id=IDS)
	shell:
		DIR+f"src/gen_custom_network.py -o "+RESDIR+"{IDS} -d "+DATABASE_PATH 


rule get_target_patterns:
	input:
		RESDIR+"{id}.arch"
	output:
		expand(RESDIR+"{id}.target",allow_missing=True)
	shell:
		DIR+"src/set_custom_target_patterns.py -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH


rule get_crosstalk:
	input:
		RESDIR+"{id}.target", RESDIR+"tf_chrom_equiv_pr_bound.out", RESDIR+"tf_pr_bound.out", RESDIR+"kpr_pr_open.out"
	output:
		RESDIR+"{id}.xtalk"
	shell:
		DIR+"src/calc_crosstalk.py -s -m "+RESDIR+" -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH+" -n 1"
		#DIR+"src/calc_crosstalk.py -s -m "+RESDIR+" -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH \
        #    + "; "+DIR+"src/calc_crosstalk.py -s -t -x -m "+RESDIR+" -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH \
		#    + "; "+DIR+"src/calc_crosstalk.py -s -c -m "+RESDIR+" -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH \
		#    + "; "+DIR+"src/calc_crosstalk.py -c -t -x -m "+RESDIR+" -i "+RESDIR+"{wildcards.id} -d "+DATABASE_PATH
