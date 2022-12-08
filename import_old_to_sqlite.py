from numpy import *

import shelve
import dill
import sqlite3

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import sys, getopt
import os
import importlib

sys.path.append('src')
from boolarr import *

# TODO:
# [x] store achieved patterns as byte streams from boolean array
# [ ] allow defining achieved patterns by # bound TFs required to count as active
# [ ] set of functions to retrieve architecture, patterns, etc. for network (using reshape)
# [ ] load into pandas dataframes?


def print_usage():
    print("usage is: test_sqlite.py -d -p <folder_in>")

def main(argv):
    populate_tables = False
    display_db = False
    folder_in = ""

    try:
        opts, args = getopt.getopt(argv,"hdp:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if len(argv) < 1:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-p":
            populate_tables = True
            folder_in = arg
        elif opt == "-d":
            display_db = True

    con = sqlite3.connect("test.db")
    cur = con.cursor()
    
    if populate_tables:
        sys.path.append(folder_in)
        sys.path.append(folder_in + "/src")
        sys.path.append(folder_in + "/res")
        

        ### GENERATE TABLES OF PARAMETERS, NETWORKS, PATTERNS, XTALK ###
        param_table_list = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters';").fetchall()
        if param_table_list == []:
            print("Parameter table not found---generating...")
            cur.execute("CREATE TABLE parameters(prefix,N_PF,N_TF,M_ENH,M_GENE,THETA,n,K_S,K_NS,n0,NUM_RANDINPUTS)")
            con.commit()
        
        network_table_list = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = 'networks';").fetchall()
        if network_table_list == []:
            print("Network table not found---generating...")
            cur.execute("CREATE TABLE networks(prefix,parameter_rowid,R,G,T)")
            con.commit()
        
        pattern_table_list = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = 'patterns';").fetchall()
        if pattern_table_list == []:
            print("Pattern table not found---generating...")
            cur.execute("CREATE TABLE patterns(network_rowid,input,achieved)")
            con.commit()
        
        xtalk_table_list = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = 'xtalk';").fetchall()
        if xtalk_table_list == []:
            print("Crosstalk table not found---generating...")
            cur.execute("CREATE TABLE xtalk(network_rowid,target_pattern,optimized_input,output_expression,fun,jac,message,nfev,nit,njev,status,success)")
            con.commit()
        
        
        ### POPULATE TABLES FROM EXISTING DATA ###
        
        prefixes = ["kpr","tf"]
        suffix = ".xtalk"
        
        for prefix in prefixes:
            cur_p = importlib.import_module(prefix + "-params")
        
            cur.execute("INSERT INTO parameters VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            [prefix,cur_p.N_PF,cur_p.N_TF,cur_p.M_ENH,cur_p.M_GENE,cur_p.THETA,cur_p.n,cur_p.K_S,cur_p.K_NS,cur_p.n0,cur_p.NUM_RANDINPUTS])
            cur_rowid = cur.lastrowid
        
            files = os.listdir(folder_in + "/res/")
            filenames = [fn for fn in files if fn.startswith(prefix) & fn.endswith(suffix)]
            filenames = [fn[:-len(suffix)] for fn in filenames]
            
            for cur_filename in filenames:
                print(cur_filename)
            
                filename_in = folder_in + "/res/" + cur_filename
                
                with shelve.open(filename_in + ".arch") as ms:
                    for key in ms:
                        globals()[key] = ms[key]
        
                # add entry to network table
                # (prefix,parameter_rowid,R,G,T)
        
                # note: to undo seralization to byte representation, use numpy.frombuffer(.)
                cur.execute("INSERT INTO networks VALUES(?,?,?,?,?)",
                            [prefix,cur_rowid,R.tobytes(),G.tobytes(),T.tobytes()])
                network_rowid = cur.lastrowid
        
            
                # POPULATE patterns #
                with shelve.open(filename_in + ".achieved") as ms:
                    for key in ms:
                        globals()[key] = ms[key]
         
                for ii, (key,value) in enumerate(mappings.items()):
                    output_bool = int2bool(key,cur_p.M_GENE)
                    for cur_val in value:
                        input_bool = int2bool(cur_val,cur_p.N_PF + cur_p.N_TF)
                        # (network_rowid,input,achieved)
                        cur.execute("INSERT INTO patterns VALUES(?,?,?)",
                                    [network_rowid,input_bool.tobytes(),
                                     output_bool.tobytes()])
        
     
                # POPULATE xtalk #
                optres = dill.load(open(filename_in + ".xtalk", "rb"))
     
                for target_pattern, cur_opt in optres.items():
                    target_pattern_bool = int2bool(target_pattern,cur_p.M_GENE)
                    cur.execute("INSERT INTO xtalk VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                [network_rowid,target_pattern_bool.tobytes(),
                                 cur_opt[0].x.tobytes(),
                                None,cur_opt[0].fun,cur_opt[0].jac.tobytes(),
                                cur_opt[0].message,cur_opt[0].nfev,cur_opt[0].nit,
                                cur_opt[0].njev,cur_opt[0].status,cur_opt[0].success])
                        
                con.commit()

    
    if display_db:
        # for ach in cur.execute("SELECT achieved FROM patterns LIMIT 5;").fetchall():
            # print(f"{frombuffer(ach[0],dtype=bool)}")

        # set_printoptions(threshold=sys.maxsize)
        for ri in cur.execute("SELECT rowid, parameter_rowid, R FROM networks WHERE prefix = 'kpr' LIMIT 5;").fetchall():
            nri = ri[0]

            params = cur.execute(f"SELECT M_ENH, N_PF FROM parameters WHERE rowid = {ri[1]};").fetchone()
            M = frombuffer(ri[2])
            M = reshape(M,[params[0],params[1]])

            achieved_pattern = cur.execute(f"SELECT achieved FROM patterns WHERE network_rowid = {nri};").fetchone()
            achieved_pattern = frombuffer(achieved_pattern[0],dtype=bool)
            print(f"N_PF = {params[1]}, achieved_pattern = {achieved_pattern}")

        
    
    con.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
