import sqlite3
import params
import os, sys
import numpy as np
import matplotlib.pyplot as plt

def extract_local_id(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])

def get_param_names():
    return [i for i in dir(params) if not(i.startswith("__"))]

def get_col_names(cur):
    return list(map(lambda x: x[0], cur.description))

# Initialize the parameter, network, pattern, and crosstalk tables in the database.
def init_tables(db_filename,db_type="child"):
    assert not(os.path.exists(db_filename)), "database already exists"
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    ### GENERATE TABLES OF PARAMETERS, NETWORKS, PATTERNS, XTALK ###

    # get parameter names
    param_list = ",".join(get_param_names())

    cur.execute(f"CREATE TABLE parameters({param_list})")
    cur.execute("CREATE TABLE networks(parameter_rowid,local_id,R,G,T)")
    if db_type == "parent":
        cur.execute("ALTER TABLE parameters ADD COLUMN folder")
    elif db_type == "child":
        pass
    else:
        print("unrecognized database type "+db_type)
        return 1

    cur.execute("CREATE TABLE patterns(network_rowid,input,target)")
    cur.execute("CREATE TABLE xtalk(network_rowid,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optimized_input,output_expression,fun,jac,message,nfev,nit,njev,status,success)")

    con.commit()
    con.close()
    return 0


# Add the current parameters (from "import params") to the database.
def add_parameters(db_filename):
    assert os.path.exists(db_filename), "error adding parameters: "+db_filename+" does not exist"

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    ncols = cur.execute("SELECT COUNT(*) FROM pragma_table_info('parameters')").fetchone()[0]
    param_names = get_param_names()

    col_insert_query = ",".join(param_names)
    param_values = [getattr(params,param) for param in param_names]
    param_insert_query = ",".join([f"{x}" for x in param_values])

    cur.execute(f"INSERT INTO parameters ({col_insert_query}) VALUES({param_insert_query})")

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

# Return network architecture with specified id.
def get_network(db_filename,local_id):
    if not(os.path.exists(db_filename)):
        print("error getting network: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    arch = cur.execute(f"SELECT R, G, T FROM networks WHERE local_id = {local_id}").fetchall()
    assert len(arch) == 1, f"{len(arch)} entries found with local_id {local_id}"

    con.close()

    R = np.frombuffer(arch[0][0])
    R = np.reshape(R,(params.M_ENH,params.N_PF))

    G = np.frombuffer(arch[0][1])
    G = np.reshape(G,(params.M_GENE,params.M_ENH))

    T = np.frombuffer(arch[0][2])
    T = np.reshape(T,(params.M_ENH,params.N_TF))
    return R, T, G


def plot_xtalk_results(database,folder_out):
    THRESH_FOR_BARPLOT = 10

    if not(os.path.exists(folder_out)):
        os.mkdir(folder_out)

    res = query_db(database,"SELECT * FROM xtalk")
    NF = query_db(database,"SELECT N_PF, N_TF, M_GENE FROM parameters")[0]

    txtoffset = 0.01
    def lab_bars(ax,prop):
        for ii in range(len(prop)):
            ax.text(ii,prop[ii]+txtoffset*max(prop),f"{prop[ii]:.3f}",ha="center")

    def get_cdf(x):
        x_sorted = sort(x)
        y = range(1,len(x_sorted)+1)
        return x_sorted, y

    for ii in range(len(res)):
        network_rowid = res[ii][0]
        target_pattern = np.frombuffer(res[ii][1])
        optimized_input = np.frombuffer(res[ii][2])
        output_expression = np.frombuffer(res[ii][3])
        xtalk = res[ii][4]

        plt.rcParams.update({'font.size':24})
        fig, (ax1,ax2) = plt.subplots(2,figsize=(24,48))

        if max(list(NF)) <= THRESH_FOR_BARPLOT:
            labs = [f"PF {p+1}" for p in range(NF[0])] + [f"TF {t+1}" for t in range(NF[1])]
            ax1.bar(labs,optimized_input)
            ax1.set_xlabel("regulatory factor")
            ax1.set_ylabel("concentration (nM)")
            lab_bars(ax1,optimized_input)

            ax2.bar([f"gene {g+1}" for g in range(NF[2])],output_expression)
            ax2.set_ylim([0,1.1])
            ax2.set_xlabel("gene")
            ax2.set_ylabel("probability expressing")
            lab_bars(ax2,output_expression)
            ax2.set_title(f"target pattern = {target_pattern}")
            lab_bars(ax1,optimized_input)
        else:
            xoi, yoi = get_cdf(optimized_input)
            ax1.plot(xoi,yoi)
            ax1.set_xlabel("optimized input concentration (nM)")
            ax2.set_ylabel("number regulatory factors")
            
            N_ON = sum(target_pattern)
            N_OFF = len(target_pattern) - N_ON
            xpe, ype = get_cdf(output_expression)
            ax2.plot(xpe,ype)
            ax2.set_xlabel("probability expressing")
            ax2.set_ylabel("number genes")
            ax2.set_title(f"target pattern = {N_ON} ON, {N_OFF} OFF")

        plt.savefig(os.path.join(folder_out,f"rowid{network_rowid}_{ii}.png"))


# Add a target pattern to the database.
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


# Return unique target patterns for network with specified id.
def get_target_patterns(db_filename,network_rowid):
    if not(os.path.exists(db_filename)):
        print("error getting target pattern: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(f"SELECT target FROM patterns WHERE network_rowid = {network_rowid}").fetchall()

    con.commit()
    con.close()

    target_patterns = [np.frombuffer(x[0]) for x in res]
    return list(map(np.array,set(map(tuple,target_patterns))))


# note: local_id is used in place of network_rowid so that it can be set directly by Snakefile
def add_xtalk(db_filename,local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optres,output_expression):
    if not(os.path.exists(db_filename)):
        print("error adding crosstalk result: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO xtalk VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern.tobytes(),
                 optres.x.tobytes(),
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

    res_table = cur.execute(f"SELECT target_pattern FROM xtalk WHERE network_rowid = {network_rowid}").fetchall()

    con.commit()
    con.close()

    patterns_already_evaluated = [np.frombuffer(x[0]) for x in res_table]

    if len(patterns_already_evaluated) > 0:
        return any([np.array_equal(target_pattern,x) for x in patterns_already_evaluated])
    else:
        return False

# Query the database.
def query_db(db_filename,query):
    if not(os.path.exists(db_filename)):
        print("error querying database: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(query).fetchall()

    con.close()
    return res

# Print the provided results formatted for the appropriate
# (specified) table.
def print_res(db_filename,table,form="short"):
    assert (form == "short") or (form == "long"), "format must be short or long"

    if not(os.path.exists(db_filename)):
        print("error querying database: "+db_filename+" does not exist")
        return 1

    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    res = cur.execute(f"SELECT * FROM {table}").fetchall()
    col_names = get_col_names(cur)

    if form == "short":
        nentry_to_display = 5
    else:
        nentry_to_display = len(res[0])


    if table == "networks":
        if not(len(res[0])) == 5:
            print("for networks, res must have 5 entries")
            return 1

        for ii in range(len(res)):
            parameter_rowid = res[ii][0]
            local_id = res[ii][1]

            cur_params = cur.execute(f"SELECT M_ENH, N_PF, M_GENE, N_TF from parameters WHERE rowid = {parameter_rowid}").fetchone()

            M_ENH = cur_params[0]
            N_PF = cur_params[1]
            M_GENE = cur_params[2]
            N_TF = cur_params[3]

            R = np.frombuffer(res[ii][2])
            R = np.reshape(R,(M_ENH,N_PF))

            G = np.frombuffer(res[ii][3])
            G = np.reshape(G,(M_GENE,M_ENH))

            T = np.frombuffer(res[ii][4])
            T = np.reshape(T,(M_ENH,N_TF))

            print(f"parameter_rowid = {parameter_rowid}, local_id = {local_id}, R = {R}, G = {G}, T = {T}")
    elif table == "patterns":
        if not(len(res[0])) == 3:
               print("for patterns, res must have 3 entries")
               return 1

        for ii in range(len(res)):
            print(f"network_rowid = {res[ii][0]}, input = {np.frombuffer(res[ii][1],dtype=bool)}, target = {np.frombuffer(res[ii][2])}")
        return 0
    elif table == "xtalk":
        if not(len(res[0])) == 15:
               print("for xtalk, res must have 15 entries")
               return 1

        for ii in range(len(res)):
            if form == "short":
                print(f"network_rowid = {res[ii][0]}, minimize_noncognate_binding = {res[ii][1]}, crosslayer_crosstalk = {res[ii][2]}, tf_first_layer = {res[ii][3]}, target_pattern = {np.frombuffer(res[ii][4])}, optimized_input = {np.frombuffer(res[ii][5])}, output_expression = {np.frombuffer(res[ii][6])}, fun = {res[ii][7]}")
            else:
                print(f"network_rowid = {res[ii][0]}, minimize_noncognate_binding = {res[ii][1]}, crosslayer_crosstalk = {res[ii][2]}, tf_first_layer = {res[ii][3]}, target_pattern = {np.frombuffer(res[ii][4])}, optimized_input = {np.frombuffer(res[ii][5])}, output_expression = {np.frombuffer(res[ii][6])}, fun = {res[ii][7]}, jac = {np.frombuffer(res[ii][8])}, message = {res[ii][9]}, nfev = {res[ii][10]}, nit = {res[ii][11]}, njev = {res[ii][12]}, status = {res[ii][13]}, success = {res[ii][14]}")
                   
    else:
        for ii in range(len(res)):
            print(", ".join(list(map(lambda x,y: f"{x} = {y}",col_names[0:nentry_to_display],res[ii][0:nentry_to_display]))))

    con.close()

    return 0


# append db to the full database db_parent
def append_db(db_parent_filename,db_filename):
    # reassign parameter_rowid and network_rowid to match default rowid assigned by SQL and
    # remove local_id column

    if not(os.path.exists(db_parent_filename)):
        print(db_parent_filename+" not found---initializing new...")
        init_tables(db_parent_filename,db_type="parent")

    db_parent = sqlite3.connect(db_parent_filename, timeout=20.0)
    parent_cur = db_parent.cursor()
    parent_cur.execute("SELECT * FROM parameters")
    parent_param_col_names = get_col_names(parent_cur)

    assert os.path.exists(db_filename), db_filename+" not found"
    
    db_child = sqlite3.connect(db_filename)
    child_cur = db_child.cursor()

    child_folder = [x for x in db_filename.split("/") if x.count("cluster_")]
    assert len(child_folder) == 1, "check child database filepath"

    parameters = child_cur.execute("SELECT * FROM parameters").fetchall()
    child_param_col_names = get_col_names(child_cur)

    print(parent_param_col_names)
    print(child_param_col_names)

    assert len(parameters) == 1, "child database must have exactly one entry in parameters table"

    if not(set(child_param_col_names).issubset(parent_param_col_names)):
        print("error: parent database has different parameter set than child database")
        return 1

    parameters = parameters[0]

    param_table = parent_cur.execute("SELECT rowid FROM parameters WHERE folder = '"+child_folder[0]+"'").fetchall()
    if len(param_table) == 0:
        col_query_str = ", ".join([f"{x}" for x in child_param_col_names])
        param_query_str = ", ".join([f"{x}" for x in parameters])
        parent_cur.execute(f"INSERT INTO parameters ({col_query_str}, folder) VALUES({param_query_str},'{child_folder[0]}')")
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
    

    ## -- APPEND PATTERNS -- ##
    local2newrowid = dict(parent_cur.execute(f"SELECT local_id, rowid FROM networks WHERE parameter_rowid = {new_parameter_rowid}").fetchall())
    pattern_data = child_cur.execute("SELECT * FROM patterns").fetchall()
    pattern_row_data = [(local2newrowid.get(a),b,c) for a,b,c in pattern_data]
    parent_cur.executemany("INSERT INTO patterns VALUES(?,?,?)",pattern_row_data)
    db_parent.commit()
    

    ## -- APPEND XTALK -- ##
    xtalk_data = child_cur.execute("SELECT * FROM xtalk").fetchall()
    xtalk_row_data = [(local2newrowid.get(a),b,c,d,e,f,g,h,i,j,k,l) for a,b,c,d,e,f,g,h,i,j,k,l in xtalk_data]
    parent_cur.executemany("INSERT INTO xtalk VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",xtalk_row_data)
    db_parent.commit()


    return 0
