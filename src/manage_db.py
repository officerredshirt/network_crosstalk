import sqlite3
import params
import os
from numpy import *

def extract_local_id(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])

# Initialize the parameter, network, pattern, and crosstalk tables in the database.
def init_tables(db_filename):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    assert not(os.path.exists("database " + db_filename + " already exists")), "database already exists"

    ### GENERATE TABLES OF PARAMETERS, NETWORKS, PATTERNS, XTALK ###
    cur.execute("CREATE TABLE parameters(N_PF,N_TF,M_ENH,M_GENE,THETA,n,K_S,K_NS,n0,NUM_RANDINPUTS)")
    cur.execute("CREATE TABLE networks(parameter_rowid,local_id,R,G,T)")
    cur.execute("CREATE TABLE patterns(network_rowid,input,achieved)")
    cur.execute("CREATE TABLE xtalk(network_rowid,target_pattern,optimized_input,output_expression,fun,jac,message,nfev,nit,njev,status,success)")

    con.commit()
    con.close()
    return 0


# Add the current parameters (from "import params") to the database.
def add_parameters(db_filename):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO parameters VALUES(?,?,?,?,?,?,?,?,?,?)",
                [params.N_PF,params.N_TF,params.M_ENH,params.M_GENE,params.THETA,params.n,params.K_S,params.K_NS,params.n0,params.NUM_RANDINPUTS])

    con.commit()
    con.close()
    return 0


# Add a network to the database.
def add_network(db_filename,local_id,R,G,T):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    nparam_entries = len(cur.execute("SELECT * FROM parameters").fetchall())
    assert nparam_entries == 1, f"parameters table has {nparam_entries} entries; must have exactly one"
    parameter_rowid = cur.execute("SELECT rowid FROM parameters").fetchone()[0]

    # note: to undo seralization to byte representation, use numpy.frombuffer(.)
    cur.execute("INSERT INTO networks VALUES(?,?,?,?,?)",
                [parameter_rowid,local_id,R.tobytes(),G.tobytes(),T.tobytes()])

    con.commit()
    con.close()
    return 0

def get_network(db_filename,local_id):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    arch = cur.execute(f"SELECT R, G, T FROM networks WHERE local_id = {local_id}").fetchall()
    assert len(arch) == 1, f"{len(arch)} entries found with local_id {local_id}"

    con.close()

    R = frombuffer(arch[0][0])
    R = reshape(R,(params.M_ENH,params.N_PF))

    G = frombuffer(arch[0][1])
    G = reshape(G,(params.M_GENE,params.M_ENH))

    T = frombuffer(arch[0][2])
    T = reshape(T,(params.M_ENH,params.N_TF))
    return R, T, G

# Add an achieved pattern to the database.
def add_pattern(db_filename,local_id,inp,out):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO patterns VALUES(?,?,?)",
                [local_id,inp.tobytes(),out.tobytes()])

    con.commit()
    con.close()
    return 0


# Returns unique achieved patterns for network with specified id.
def get_achieved_patterns(db_filename,network_rowid):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(f"SELECT achieved FROM patterns WHERE network_rowid = {network_rowid}").fetchall()

    con.commit()
    con.close()

    achieved_patterns = [frombuffer(x[0]) for x in res]
    return list(map(array,set(map(tuple,achieved_patterns))))


# note: local_id is used in place of network_rowid so that it can be set directly by Snakefile
def add_xtalk(db_filename,local_id,target_pattern,optres,output_expression):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO xtalk VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [local_id,target_pattern.tobytes(),optres.x.tobytes(),
                 output_expression.tobytes(),optres.fun,optres.jac.tobytes(),
                 optres.message,optres.nfev,optres.nit,optres.njev,
                 optres.status,optres.success])

    con.commit()
    con.close()
    return 0

# Returns True if a crosstalk result has already been calculated
# for the given network and target pattern.
def xtalk_result_found(db_filename,network_rowid,target_pattern):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res_table = cur.execute(f"SELECT * FROM xtalk WHERE network_rowid = {network_rowid}").fetchall()

    con.commit()
    con.close()

    patterns_already_evaluated = [frombuffer(x[1]) for x in res_table]

    if len(patterns_already_evaluated) > 0:
        return any([array_equal(target_pattern,x) for x in patterns_already_evaluated])
    else:
        return False


def query_db(database,query):
    con = sqlite3.connect(database)
    cur = con.cursor()

    res = cur.execute(query).fetchall()

    con.close()
    return res


def print_res(database,table,res):

    if table == "parameters":
        if not(len(res[0])) == 10:
               print("for parameters, res must have 10 entries")
               return 1

        for ii in range(len(res)):
            print(f"N_PF = {res[ii][0]}, N_TF = {res[ii][1]}, M_ENH = {res[ii][2]}, M_GENE = {res[ii][3]}, THETA = {res[ii][4]}, n = {res[ii][5]}, K_S = {res[ii][6]}, K_NS = {res[ii][7]}, n0 = {res[ii][8]}, NUM_RANDINPUTS = {res[ii][9]}")

    elif table == "networks":
        if not(len(res[0])) == 5:
               print("for networks, res must have 5 entries")
               return 1

        for ii in range(len(res)):
               parameter_rowid = res[ii][0]
               local_id = res[ii][1]

               R = frombuffer(res[ii][2])
               R = reshape(R,(params.M_ENH,params.N_PF))

               G = frombuffer(res[ii][3])
               G = reshape(G,(params.M_GENE,params.M_ENH))

               T = frombuffer(res[ii][4])
               T = reshape(T,(params.M_ENH,params.N_TF))

               print(f"parameter_rowid = {parameter_rowid}, local_id = {local_id}, R = {R}, G = {G}, T = {T}")
    elif table == "patterns":
        if not(len(res[0])) == 3:
               print("for patterns, res must have 3 entries")
               return 1

        for ii in range(len(res)):
            print(f"network_rowid = {res[ii][0]}, input = {frombuffer(res[ii][1],dtype=bool)}, achieved = {frombuffer(res[ii][2])}")
        return 0
    elif table == "xtalk":
        if not(len(res[0])) == 4 and not(len(res[0])) == 12:
               print("for xtalk, res must have 4 or 12 entries")
               return 1

        for ii in range(len(res)):
            if len(res[0]) == 4:
                print(f"network_rowid = {res[ii][0]}, target_pattern = {frombuffer(res[ii][1])}, optimized_input = {frombuffer(res[ii][2])}, output_expression = {frombuffer(res[ii][3])}")
            else:
                print(f"network_rowid = {res[ii][0]}, target_pattern = {frombuffer(res[ii][1])}, optimized_input = {frombuffer(res[ii][2])}, output_expression = {frombuffer(res[ii][3])}, fun = {res[ii][4]}, jac = {frombuffer(res[ii][5])}, message = "+res[ii][6]+f", nfev = {res[ii][7]}, nit = {res[ii][8]}, njev = {res[ii][9]}, status = {res[ii][10]}, success = {res[ii][11]}")
                   
        return 0

    return 1


# append db to the full database db_parent
def append_db(db_parent_filename,db_filename):
    # reassign parameter_rowid and network_rowid to match default rowid assigned by SQL and
    # remove local_id column or set to NULL in the full table (should NOT be used to index)

    # also add columns that store the datetime of db (filename sans path) in db_parent
    return 0
