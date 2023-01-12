import sqlite3
import params
import os
from numpy import *

def extract_local_id(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])

# Initialize the parameter, network, pattern, and crosstalk tables in the database.
def init_tables(db_filename,db_type="child"):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    assert not(os.path.exists("database " + db_filename + " already exists")), "database already exists"

    ### GENERATE TABLES OF PARAMETERS, NETWORKS, PATTERNS, XTALK ###
    cur.execute("CREATE TABLE parameters(N_PF,N_TF,M_ENH,M_GENE,THETA,n,K_S,K_NS,n0,NUM_RANDINPUTS)")
    cur.execute("CREATE TABLE networks(parameter_rowid,local_id,R,G,T)")
    if db_type == "parent":
        cur.execute("ALTER TABLE parameters ADD COLUMN folder")
    else:
        print("unrecognized database type "+db_type)
        return 1

    cur.execute("CREATE TABLE patterns(network_rowid,input,achieved)")
    cur.execute("CREATE TABLE xtalk(network_rowid,target_pattern,optimized_input,output_expression,fun,jac,message,nfev,nit,njev,status,success)")

    con.commit()
    con.close()
    return 0


# Add the current parameters (from "import params") to the database.
def add_parameters(db_filename):
    assert os.path.exists(db_filename), "error adding parameters: "+db_filename+" does not exist"

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO parameters VALUES(?,?,?,?,?,?,?,?,?,?)",
                [params.N_PF,params.N_TF,params.M_ENH,params.M_GENE,params.THETA,params.n,params.K_S,params.K_NS,params.n0,params.NUM_RANDINPUTS])

    con.commit()
    con.close()
    return 0


# Add a network to the database.
def add_network(db_filename,local_id,R,G,T):
    if not(os.path.exists(db_filename)):
        print("error adding network: "+db_filename+" does not exist")
        return 1

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
    if not(os.path.exists(db_filename)):
        print("error getting network: "+db_filename+" does not exist")
        return 1

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
    if not(os.path.exists(db_filename)):
        print("error adding pattern: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO patterns VALUES(?,?,?)",
                [local_id,inp.tobytes(),out.tobytes()])

    con.commit()
    con.close()
    return 0


# Returns unique achieved patterns for network with specified id.
def get_achieved_patterns(db_filename,network_rowid):
    if not(os.path.exists(db_filename)):
        print("error getting achieved pattern: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(f"SELECT achieved FROM patterns WHERE network_rowid = {network_rowid}").fetchall()

    con.commit()
    con.close()

    achieved_patterns = [frombuffer(x[0]) for x in res]
    return list(map(array,set(map(tuple,achieved_patterns))))


# note: local_id is used in place of network_rowid so that it can be set directly by Snakefile
def add_xtalk(db_filename,local_id,target_pattern,optres,output_expression):
    if not(os.path.exists(db_filename)):
        print("error adding crosstalk result: "+db_filename+" does not exist")
        return 1

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
    if not(os.path.exists(db_filename)):
        print("error finding crosstalk result: "+db_filename+" does not exist")
        return 1

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


def query_db(db_filename,query):
    if not(os.path.exists(db_filename)):
        print("error querying database: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(query).fetchall()

    con.close()
    return res


def print_res(table,res,form="short",db_type="child"):
    assert (form == "short") or (form == "long"), "format must be short or long"
    assert (db_type == "child") or (db_type == "parent"), "db_type must be child or parent"

    if table == "parameters":
        if db_type == "child":
            if not(len(res[0])) == 10:
                print("for parameters (db type child), res must have 10 entries")
                return 1
        else:
            if not(len(res[0])) == 11:
                print("for parameters (db type parent), res must have 11 entries")
                return 1

        for ii in range(len(res)):
            if form == "short":
                print(f"N_PF = {res[ii][0]}, N_TF = {res[ii][1]}, M_ENH = {res[ii][2]}, M_GENE = {res[ii][3]}, THETA = {res[ii][4]}, n = {res[ii][5]}")
            else:
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
        if not(len(res[0])) == 12:
               print("for xtalk, res must have 12 entries")
               return 1

        for ii in range(len(res)):
            if form == "short":
                print(f"network_rowid = {res[ii][0]}, target_pattern = {frombuffer(res[ii][1])}, optimized_input = {frombuffer(res[ii][2])}, output_expression = {frombuffer(res[ii][3])}")
            else:
                print(f"network_rowid = {res[ii][0]}, target_pattern = {frombuffer(res[ii][1])}, optimized_input = {frombuffer(res[ii][2])}, output_expression = {frombuffer(res[ii][3])}, fun = {res[ii][4]}, jac = {frombuffer(res[ii][5])}, message = "+res[ii][6]+f", nfev = {res[ii][7]}, nit = {res[ii][8]}, njev = {res[ii][9]}, status = {res[ii][10]}, success = {res[ii][11]}")
                   
        return 0

    return 1


# append db to the full database db_parent
def append_db(db_parent_filename,db_filename):
    # reassign parameter_rowid and network_rowid to match default rowid assigned by SQL and
    # remove local_id column

    if not(os.path.exists(db_parent_filename)):
        print(db_parent_filename+" not found---initializing new...")
        init_tables(db_parent_filename,db_type="parent")

    db_parent = sqlite3.connect(db_parent_filename, timeout=20.0)
    parent_cur = db_parent.cursor()

    assert os.path.exists(db_filename), db_filename+" not found"
    
    db_child = sqlite3.connect(db_filename)
    child_cur = db_child.cursor()

    child_folder = [x for x in db_filename.split("/") if x.count("cluster_")]
    assert len(child_folder) == 1, "check child database filepath"

    parameters = child_cur.execute("SELECT * FROM parameters").fetchall()
    assert len(parameters) == 1, "child database must have exactly one entry in parameters table"
    assert len(parameters[0]) == 10, "child database should have 10 columns in parameters table"
    parameters = parameters[0]

    param_table = parent_cur.execute("SELECT rowid FROM parameters WHERE folder = '"+child_folder[0]+"'").fetchall()
    if len(param_table) == 0:
        parent_cur.execute(f"INSERT INTO parameters VALUES({parameters[0]}, {parameters[1]}, {parameters[2]}, {parameters[3]}, {parameters[4]}, {parameters[5]}, {parameters[6]}, {parameters[7]}, {parameters[8]}, {parameters[9]}, '"+child_folder[0]+"')")
        new_parameter_rowid = parent_cur.lastrowid
        db_parent.commit()
    else:
        print("error: already found parameter table in association with folder "+child_folder[0])
        return 1

    ## -- APPEND NETWORKS -- ##
    networks = child_cur.execute("SELECT * FROM networks").fetchall()
    assert len(networks) > 0, "no network entries found in child database"
    assert len(networks[0]) == 5, "child database should have 5 columns in networks table"
    
    # insert entries from child into parent (use single call, not loop)
    print(f"Inserting {len(networks)} network(s) into database...")
    network_row_data = [(new_parameter_rowid,b,c,d,e) for a,b,c,d,e in networks]
    parent_cur.executemany("INSERT INTO networks VALUES(?,?,?,?,?)",network_row_data)
    db_parent.commit()
    
    """
    test = parent_cur.execute("SELECT * FROM networks").fetchall()
    assert len(test) > 0, "no results in networks table"
    print_res("networks",test,db_type="parent")
    """


    ## -- APPEND PATTERNS -- ##
    local2newrowid = dict(parent_cur.execute(f"SELECT local_id, rowid FROM networks WHERE parameter_rowid = {new_parameter_rowid}").fetchall())
    pattern_data = child_cur.execute("SELECT * FROM patterns").fetchall()
    pattern_row_data = [(local2newrowid.get(a),b,c) for a,b,c in pattern_data]
    parent_cur.executemany("INSERT INTO patterns VALUES(?,?,?)",pattern_row_data)
    db_parent.commit()

    """
    test = parent_cur.execute("SELECT * FROM patterns").fetchall()
    assert len(test) > 0, "no results in patterns table"
    print("----CHILD----")
    print_res("patterns",pattern_data,db_type="child")
    print("----PARENT----")
    print_res("patterns",test,db_type="parent")
    """
    

    ## -- APPEND XTALK -- ##
    xtalk_data = child_cur.execute("SELECT * FROM xtalk").fetchall()
    xtalk_row_data = [(local2newrowid.get(a),b,c,d,e,f,g,h,i,j,k,l) for a,b,c,d,e,f,g,h,i,j,k,l in xtalk_data]
    parent_cur.executemany("INSERT INTO xtalk VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",xtalk_row_data)
    db_parent.commit()


    return 0
