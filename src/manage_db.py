#!/usr/bin/env python3

import sqlite3
import pprint
import params
import os, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import dill
import itertools

def check_db_exists(db_filename):
    if not(os.path.exists(db_filename)):
        print("error getting network: "+db_filename+" does not exist")
        sys.exit()
    return 1

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
    cur.execute("CREATE TABLE xtalk(network_rowid,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optimized_input,output_expression,output_error,max_expression,fun,jac,message,nfev,nit,njev,status,success)")

    con.commit()
    con.close()
    return 0


# Add the current parameters (from "import params") to the database.
def add_parameters(db_filename):
    check_db_exists(db_filename)

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
    check_db_exists(db_filename)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    nparam_entries = len(cur.execute("SELECT * FROM parameters").fetchall())
    assert nparam_entries == 1, f"parameters table has {nparam_entries} entries; must have exactly one"
    parameter_rowid = cur.execute("SELECT rowid FROM parameters").fetchone()[0]

    # note: to undo seralization to byte representation, use numpy.frombuffer(.)
    cur.execute("INSERT INTO networks(parameter_rowid,local_id,R,G,T) VALUES(?,?,?,?,?)",
                [parameter_rowid,local_id,R.tobytes(),G.tobytes(),T.tobytes()])

    con.commit()
    con.close()
    return 0

# Return network architecture with specified id.
def get_network(db_filename,local_id):
    check_db_exists(db_filename)

    arch = get_formatted(db_filename,"networks",query=f"SELECT R, G, T, parameter_rowid FROM networks WHERE local_id = {local_id}")
    assert len(arch) == 1, f"{len(arch)} entries found with local_id {local_id}"

    return arch[0]["R"], arch[0]["T"], arch[0]["G"]


# Return results for the specified table formatted appropriately.
def get_formatted(db_filename,table,query=None):
    check_db_exists(db_filename)

    if query == None:
        query = f"SELECT * FROM {table}"

    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    res = cur.execute(query).fetchall()
    col_names = get_col_names(cur)

    formatted_res = [{}]*len(res)
    for ii in range(len(res)):
        formatted_res[ii] = {col_names[jj]: res[ii][jj] for jj in range(len(res[ii]))}

        if table == "patterns":
            try:
                formatted_res[ii]["input"] = np.frombuffer(formatted_res[ii]["input"],dtype=bool)
            except:
                pass

            try:
                formatted_res[ii]["target"] = np.frombuffer(formatted_res[ii]["target"])
            except:
                pass
        elif table == "networks":
            cur_params = cur.execute(f"SELECT M_ENH, N_PF, M_GENE, N_TF from parameters WHERE rowid = {formatted_res[ii]['parameter_rowid']}").fetchone()
            M_ENH = cur_params[0]
            N_PF = cur_params[1]
            M_GENE = cur_params[2]
            N_TF = cur_params[3]

            try:
                formatted_res[ii]["R"] = np.reshape(np.frombuffer(formatted_res[ii]["R"]),(M_ENH,N_PF))
            except:
                pass
            try:
                formatted_res[ii]["G"] = np.reshape(np.frombuffer(formatted_res[ii]["G"]),(M_GENE,M_ENH))
            except:
                pass
            try:
                formatted_res[ii]["T"] = np.reshape(np.frombuffer(formatted_res[ii]["T"]),(M_ENH,N_TF))
            except:
                pass
        elif table == "xtalk":
            try:
                formatted_res[ii]["target_pattern"] = np.frombuffer(formatted_res[ii]["target_pattern"])
            except:
                pass
            try:
                formatted_res[ii]["optimized_input"] = np.frombuffer(formatted_res[ii]["optimized_input"])
            except:
                pass
            try:
                formatted_res[ii]["output_expression"] = np.frombuffer(formatted_res[ii]["output_expression"])
            except:
                pass
            try:
                formatted_res[ii]["output_error"] = np.frombuffer(formatted_res[ii]["output_error"])
                formatted_res[ii]["output_error"] = np.reshape(formatted_res[ii]["output_error"],
                                                           (round(len(formatted_res[ii]["output_error"])/3),3))
            except:
                pass
            try:
                formatted_res[ii]["jac"] = np.frombuffer(formatted_res[ii]["jac"])
            except:
                pass
            try:
                formatted_res[ii]["layer2_cS_for_fixed_other_cNS"] = np.frombuffer(formatted_res[ii]["layer2_cS_for_fixed_other_cNS"])
            except:
                pass

    con.close()
    return formatted_res


# idealized curve: given a concentration of noncogate factors,
# what specific layer 2 concentration would give exactly the
# target expression level? (ignoring layer 1 binding)
def calc_optimal_cS_for_fixed_other_cNS(db_filename):
    check_db_exists(db_filename)

    # add column to database if doesn't exist
    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    if "layer2_cS_for_fixed_other_cNS" not in [x[1] for x in cur.execute("PRAGMA table_info(xtalk)")]:
        cur.execute("ALTER TABLE xtalk ADD COLUMN layer2_cS_for_fixed_other_cNS")

    res = get_formatted(db_filename,"xtalk")

    def objective_fn(c_NS,c_S,target):
        return tf_pr_bound(c_NS+c_S,c_S) - target

    for ii, target in enumerate(res["target_pattern"]):
        if res["layer2_cS_for_fixed_other_cNS"][ii] == None:
            layer2_cS_given_noncog = np.zeros(len(res["target_pattern"][0]))
            for ii_gene, target_level in enumerate(target):
                cur_C_NS = np.sum(res["optimized_input"][ii]) - res["optimized_input"][ii][ii_gene]
                layer2_cS_given_noncog[ii_gene] = scipy.optimize.fsolve(lambda x: objective_fn(cur_C_NS,x,target),res["optimized_input"][ii][ii_gene])
        cur.execute(f"UPDATE xtalk SET layer2_cS_for_fixed_other_cNS = {layer2_cS_given_noncog.tobytes()} WHERE network_rowid = {res['network_rowid'][ii]}")

    con.close()
    #TODO: back up databases, test this function, add to compile_results, use in plotting


def plot_xtalk_errors(db_folder,folder_out):
    plot_error_contributions(os.path.join(db_folder,"local_db.db"),folder_out)
    plot_error_fraction(db_folder,folder_out)
    return 0


# Return elements of arr where corresponding index in ixs is true.
def logical_ix(arr,ixs):
    if not len(arr) == len(ixs):
        print("error: for logical indexing arr must be same length as ixs")
        return
    return list(itertools.compress(arr,ixs))


# Scatterplot where "valid" results are green and "invalid" results are red.
def scatter_valid(ax,x,y,val_ixs,lab="data"):
    x_valid = logical_ix(x,val_ixs)
    x_invalid = logical_ix(x,np.invert(val_ixs))

    y_valid = logical_ix(y,val_ixs)
    y_invalid = logical_ix(y,np.invert(val_ixs))

    ax.scatter(x_invalid,y_invalid,c="r",label=lab)
    ax.scatter(x_valid,y_valid,c="g",label=lab)


def plot_error_contributions(db_filename,folder_out):
    check_db_exists(db_filename)

    if not(os.path.exists(folder_out)):
        os.mkdir(folder_out)

    res = get_formatted(db_filename,"xtalk")

    patterr_res = logical_ix(res,[x["minimize_noncognate_binding"] == 0 for x in res])
    if len(patterr_res) == 0:
        print("no results found using patterning error as metric")
        return 1

    network_rowids = [x["network_rowid"] for x in patterr_res]
    unique_rowids = np.unique(network_rowids)
    
    plt.rcParams.update({'font.size':24})

    for cur_rowid in unique_rowids:
        cur_res = logical_ix(patterr_res,np.isin(network_rowids,cur_rowid))
        num_cur_res = len(cur_res)

        if num_cur_res > 0:
            cur_target_patterns, cur_total_error_frac, cur_output_expression, _, _, _, cur_layer, _, _, _, _, on_ix, off_ix = compile_results(cur_res)

            tf_only_ix = np.repeat(cur_layer==1,len(cur_res[0]["target_pattern"]))
            kpr_ix = np.repeat(cur_layer==0,len(cur_res[0]["target_pattern"]))

            cur_target_patterns_on_tf = logical_ix(cur_target_patterns,np.logical_and(on_ix,tf_only_ix))
            cur_target_patterns_on_kpr = logical_ix(cur_target_patterns,np.logical_and(on_ix,kpr_ix))

            on_expression = logical_ix(cur_output_expression,on_ix)
            off_error_frac = logical_ix(cur_total_error_frac,off_ix)

            metric_abs_errs = cur_output_expression - cur_target_patterns
            metric_abs_errs = metric_abs_errs*np.abs(metric_abs_errs) # preserve sign

            metric_abs_errs_on_tf = np.array(logical_ix(metric_abs_errs,np.logical_and(on_ix,tf_only_ix)))
            metric_abs_errs_off_tf = np.array(logical_ix(metric_abs_errs,np.logical_and(off_ix,tf_only_ix)))
            metric_abs_errs_on_kpr = np.array(logical_ix(metric_abs_errs,np.logical_and(on_ix,kpr_ix)))
            metric_abs_errs_off_kpr = np.array(logical_ix(metric_abs_errs,np.logical_and(off_ix,kpr_ix)))


            fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(45,24))

            ax1.boxplot((metric_abs_errs_on_tf,abs(metric_abs_errs_on_tf),metric_abs_errs_on_kpr,abs(metric_abs_errs_on_kpr)),labels=("TF (signed)","TF","chromatin (signed)","chromatin"))
            ax1.set_title(f"patterning error terms, ON genes",wrap=True)

            ax2.boxplot((metric_abs_errs_off_tf,metric_abs_errs_off_kpr),labels=("TF","chromatin"))
            ax2.set_title(f"patterning error terms, OFF genes",wrap=True)

            ax3.scatter(cur_target_patterns_on_tf,metric_abs_errs_on_tf,label="TF")
            ax3.scatter(cur_target_patterns_on_kpr,metric_abs_errs_on_kpr,label="chromatin")
            ax3.set_xlabel("target expression")
            ax3.set_ylabel("patterning error term (signed)")
            ax3.legend()
            
            plt.savefig(os.path.join(folder_out,f"patterning_error_contributions_rowid{cur_rowid}.png")) 


def load_pr_on(db_folder):
    tf_chrom_equiv_pr_bound = dill.load(open(os.path.join(db_folder,"tf_chrom_equiv_pr_bound.out"),"rb"))
    kpr_pr_open = dill.load(open(os.path.join(db_folder,"kpr_pr_open.out"),"rb"))
    tf_pr_bound = dill.load(open(os.path.join(db_folder,"tf_pr_bound.out"),"rb"))

    return (tf_chrom_equiv_pr_bound,kpr_pr_open,tf_pr_bound)


def compile_results(cur_res,N_PF=[],N_TF=[],R=[],G=[]):
    val_res_ix = [x["success"] == 1 for x in cur_res]
    num_cur_res = len(cur_res)

    cur_target_patterns = np.empty(num_cur_res,dtype=object)
    cur_total_error_frac = np.empty(num_cur_res,dtype=object)
    cur_output_expression = np.empty(num_cur_res,dtype=object)
    cur_optimized_input = np.empty(num_cur_res,dtype=object)
    cur_cluster_mean_expression = np.empty(num_cur_res,dtype=object)
    cur_metric = np.empty(num_cur_res)
    cur_layer = np.empty(num_cur_res)
    cur_mean_on_expression = np.empty(num_cur_res)
    for ii, cur_entry in enumerate(cur_res):
        cur_target_patterns[ii] = cur_entry["target_pattern"]
        cur_total_error_frac[ii] = cur_entry["output_error"][:,2]
        cur_output_expression[ii] = cur_entry["output_expression"]
        cur_optimized_input[ii] = cur_entry["optimized_input"]
        cur_metric[ii] = cur_entry["fun"]
        cur_mean_on_expression[ii] = np.mean(logical_ix(cur_output_expression[ii],cur_target_patterns[ii] > 0))
        cur_layer[ii] = cur_entry["tf_first_layer"]

        if not G == [] and not R == [] and not N_PF == []:
            # gene to PF mapping
            gene_to_pf = np.matmul(G,R)
            cur_cluster_mean_expression[ii] = np.empty(N_PF,dtype=object)
            for ii_cluster in range(N_PF):
                cur_cluster_mean_expression[ii][ii_cluster] = np.mean(logical_ix(cur_target_patterns[ii],gene_to_pf[:,ii_cluster] == 1))
        elif ii == 0:
            print("no R, G, N_PF provided; skipping current cluster mean expression")

    if not N_PF == [] and not N_TF == []:
        pf_ix = [False]*(N_TF+N_PF)
        pf_ix[0:N_PF] = [True]*N_PF
        tf_ix = np.invert(pf_ix)
    else:
        print("no N_TF, N_PF provided; skipping pf_ix, tf_ix")
        pf_ix = []
        tf_ix = []

    cur_target_patterns = np.concatenate(tuple(cur_target_patterns))
    cur_total_error_frac = np.concatenate(tuple(cur_total_error_frac))
    cur_output_expression = np.concatenate(tuple(cur_output_expression))
    cur_optimized_input = np.concatenate(tuple(cur_optimized_input))
    try:
        cur_cluster_mean_expression = np.concatenate(tuple(cur_cluster_mean_expression))
    except:
        pass

    on_ix = cur_target_patterns > 0
    off_ix = cur_target_patterns == 0

    return cur_target_patterns, cur_total_error_frac, cur_output_expression, cur_optimized_input, cur_cluster_mean_expression, cur_metric, cur_layer, cur_mean_on_expression, val_res_ix, pf_ix, tf_ix, on_ix, off_ix


# generates plots of error fraction for all ON vs. OFF genes pooled across all target patterns
# NOTE: assumes one-to-one mapping from Layer 2 factors to target genes
def plot_error_fraction(db_folder,folder_out):
    db_filename = os.path.join(db_folder,"local_db.db")
    check_db_exists(db_filename)

    (tf_chrom_equiv_pr_bound,kpr_pr_open,tf_pr_bound) = load_pr_on(db_folder)

    if not(os.path.exists(folder_out)):
        os.mkdir(folder_out)

    res = get_formatted(db_filename,"xtalk")

    network_rowids = [x["network_rowid"] for x in res]
    unique_rowids = np.unique(network_rowids)

    plt.rcParams.update({'font.size':24})
    # filter by optimization strategy, first layer (TF or chromatin), and rowid
    for metric in [0,1]:
        if metric:
            metric_label = "noncognate binding error"
            metric_filename = "noncognate"
        else:
            metric_label = "patterning error"
            metric_filename = "patterning"
        m_res = logical_ix(res,[x["minimize_noncognate_binding"] == metric for x in res])
        if len(m_res) > 0:
            for first_layer in [0,1]:
                if first_layer:
                    layer_label = "TF"
                    layer_filename = "tf"
                else:
                    layer_label = "chromatin"
                    layer_filename = "kpr"
                fl_res = logical_ix(m_res,[x["tf_first_layer"] == first_layer for x in m_res])
                if len(fl_res) > 0:
                    cur_network_rowids = [x["network_rowid"] for x in fl_res]
                    for cur_rowid in unique_rowids:
                        parameter_rowid = query_db(db_filename,f"SELECT parameter_rowid FROM networks WHERE local_id = {cur_rowid}")[0][0]
                        (N_PF,N_TF) = query_db(db_filename,f"SELECT N_PF, N_TF from parameters WHERE rowid = {parameter_rowid}")[0]
                        (R,T,G) = get_network(db_filename,cur_rowid)

                        # compile patterns sharing the same network_rowid
                        cur_res = logical_ix(fl_res,np.isin(cur_network_rowids,cur_rowid))
                        num_cur_res = len(cur_res)
                        cur_target_patterns, cur_total_error_frac, cur_output_expression, cur_optimized_input, cur_cluster_mean_expression, cur_metric, cur_layer, cur_mean_on_expression, val_res_ix, pf_ix, tf_ix, on_ix, off_ix = compile_results(cur_res,N_PF,N_TF,R,G)

                        on_expression = logical_ix(cur_output_expression,on_ix)
                        on_error_frac = logical_ix(cur_total_error_frac,on_ix)
                        on_noncog_exp = np.array(on_error_frac)*np.array(on_expression)

                        off_error_frac = logical_ix(cur_total_error_frac,off_ix)
                        off_expression = logical_ix(cur_output_expression,off_ix)

                        pf_ix = np.tile(pf_ix,len(cur_res))
                        pf_optimized_input = np.array(logical_ix(cur_optimized_input,pf_ix))
                        tf_optimized_input = np.array(logical_ix(cur_optimized_input,np.invert(pf_ix)))

                        total_pf_concentration_per_pf_per_pattern = np.tile(np.sum(pf_optimized_input.reshape(len(cur_res),N_PF),axis=1),N_PF)
                        total_tf_concentration_per_tf_per_pattern = np.tile(np.sum(tf_optimized_input.reshape(len(cur_res),N_TF),axis=1),N_TF)
                        
                        if layer_filename == "kpr":
                            pf_optimized_pr_on = np.array(list(map(kpr_pr_open,total_pf_concentration_per_pf_per_pattern,pf_optimized_input)))/cur_res[0]["max_expression"]
                        else:
                            pf_optimized_pr_on = np.array(list(map(tf_chrom_equiv_pr_bound,total_pf_concentration_per_pf_per_pattern,pf_optimized_input)))
                        tf_optimized_pr_bound = np.array(list(map(tf_pr_bound,total_tf_concentration_per_tf_per_pattern,tf_optimized_input)))


                        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(45,36))

                        filename = f"error_fraction_rowid{cur_rowid}_{metric_filename}_{layer_filename}"
                        title_str = f"rowid {cur_rowid}: ON error fraction, {num_cur_res} target patterns ({layer_label} first layer)"

                        val_pat_ix = np.repeat(val_res_ix,round(len(cur_target_patterns)/num_cur_res))
                        ax1.plot([0,1],[0,1],color="gray",linewidth=5)
                        scatter_valid(ax1,cur_target_patterns,cur_output_expression,val_pat_ix)
                        ax1.set_xlabel("target expression level")
                        ax1.set_ylabel("actual expression level")

                        scatter_valid(ax2,logical_ix(cur_target_patterns,on_ix),
                                    on_error_frac,logical_ix(val_pat_ix,on_ix))
                        ax2.set_xlabel("target expression level")
                        ax2.set_ylabel("total error fraction (ON genes)")

                        noncog_expression = cur_output_expression*cur_total_error_frac
                        scatter_valid(ax3,cur_target_patterns,noncog_expression,val_pat_ix)
                        ax3.set_xlabel("target expression level")
                        ax3.set_ylabel("total expression level * error fraction")

                        scatter_valid(ax4,cur_mean_on_expression,cur_metric,val_res_ix)
                        ax4.set_xlabel("mean expression of ON genes")
                        ax4.set_ylabel(metric_label)

                        val_cluster_ix = np.repeat(val_res_ix,round(len(cur_cluster_mean_expression)/num_cur_res))
                        scatter_valid(ax5,cur_cluster_mean_expression,pf_optimized_input,val_cluster_ix)
                        ax5.set_xlabel("mean target expression in cluster")
                        ax5.set_ylabel("layer 1 concentration for corresponding cluster")

                        tf_sweep = np.linspace(min(tf_optimized_input),max(tf_optimized_input),100)
                        layer2_induction_no_xtalk = np.array(list(map(tf_pr_bound,tf_sweep,tf_sweep)))
                        ax6.plot(layer2_induction_no_xtalk,tf_sweep,color="gray",linewidth=5)
                        scatter_valid(ax6,cur_target_patterns,tf_optimized_input,val_pat_ix)
                        ax6.set_xlabel("target expression level")
                        ax6.set_ylabel("layer 2 TF concentration for corresponding target")

                        # idealized curve: given a concentration of noncogate factors,
                        # what specific layer 2 concentration would give exactly the
                        # target expression level? (ignoring layer 1 binding)

                        def objective_fn(c_NS,c_S,target):
                            return tf_pr_bound(c_NS+c_S,c_S) - target

                        # solve for C_S with fixed C_NS s.t. tf_pr_bound = target_expression
                        layer2_c_S_given_noncog = np.zeros(len(cur_target_patterns))
                        for ii, ii_target in enumerate(cur_target_patterns):
                            cur_c_NS = total_tf_concentration_per_tf_per_pattern[ii] - tf_optimized_input[ii]
                            layer2_c_S_given_noncog[ii] = scipy.optimize.fsolve(lambda x: objective_fn(cur_c_NS,x,ii_target),tf_optimized_input[ii])

                        ax7.plot(layer2_induction_no_xtalk,tf_sweep,color="gray",linewidth=5)
                        ax7.scatter(cur_target_patterns,tf_optimized_input,c="b")
                        scatter_valid(ax7,cur_target_patterns,layer2_c_S_given_noncog,val_pat_ix)
                        ax7.set_xlabel("target expression level")
                        ax7.set_ylabel("layer 2 TF concentration given noncognate concentration")

                        scatter_valid(ax8,cur_cluster_mean_expression,pf_optimized_pr_on,val_cluster_ix)
                        ax8.set_xlabel("mean target expression in cluster")
                        ax8.set_ylabel("probability layer 1 on")

                        ax9.plot([0,1],[0,1],color="gray",linewidth=5)
                        scatter_valid(ax9,cur_target_patterns,tf_optimized_pr_bound,val_pat_ix)
                        ax9.set_xlabel("target expression level")
                        ax9.set_ylabel("probability layer 2 bound")

                        plt.savefig(os.path.join(folder_out,f"{filename}.png")) 
    plt.close("all")


def plot_xtalk_results(database,folder_out):
    THRESH_FOR_BARPLOT = 10

    if not(os.path.exists(folder_out)):
        os.mkdir(folder_out)

    res = query_db(database,"SELECT network_rowid, target_pattern, optimized_input, output_expression, xtalk FROM xtalk")
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
        plt.close("all")


# Add a target pattern to the database.
def add_pattern(db_filename,local_id,inp,out):
    check_db_exists(db_filename)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO patterns VALUES(?,?,?)",
                [local_id,inp.tobytes(),out.tobytes()])

    con.commit()
    con.close()
    return 0


# Return unique target patterns for network with specified id.
def get_target_patterns(db_filename,network_rowid):
    check_db_exists(db_filename)
    res = get_formatted(db_filename,"patterns",f"SELECT input, target FROM patterns WHERE network_rowid = {network_rowid}")
    return [x["input"] for x in res], [x["target"] for x in res]


# note: local_id is used in place of network_rowid so that it can be set directly by Snakefile
def add_xtalk(db_filename,local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optres,output_expression,output_error,max_expression):
    check_db_exists(db_filename)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("INSERT INTO xtalk (network_rowid,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optimized_input,output_expression,output_error,max_expression,fun,jac,message,nfev,nit,njev,status,success) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,
                 target_pattern.tobytes(),
                 optres.x.tobytes(),
                 output_expression.tobytes(),output_error.tobytes(),
                 max_expression,
                 optres.fun,optres.jac.tobytes(),
                 optres.message,optres.nfev,optres.nit,optres.njev,
                 optres.status,optres.success])

    con.commit()
    con.close()
    return 0

# Returns True if a crosstalk result has already been calculated
# for the given network, target pattern, patterning metric, and
# first layer (chromatin or TF).
def xtalk_result_found(db_filename,network_rowid,minimize_noncognate_binding,tf_first_layer,target_pattern):
    check_db_exists(db_filename)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res_table = cur.execute(f"SELECT target_pattern FROM xtalk WHERE network_rowid = {network_rowid} AND minimize_noncognate_binding = {minimize_noncognate_binding} AND tf_first_layer = {tf_first_layer}").fetchall()

    con.commit()
    con.close()

    patterns_already_evaluated = [np.frombuffer(x[0]) for x in res_table]

    if len(patterns_already_evaluated) > 0:
        return any([np.array_equal(target_pattern,x) for x in patterns_already_evaluated])
    else:
        return False


# Query the database.
def query_db(db_filename,query):
    check_db_exists(db_filename)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    res = cur.execute(query).fetchall()

    con.close()
    return res


# Print the provided results formatted for the appropriate
# (specified) table.
def print_res(db_filename,table,form="short"):
    assert (form == "short") or (form == "long"), "format must be short or long"
    check_db_exists(db_filename)

    res = get_formatted(db_filename,table)

    for ii in range(len(res)):
        pprint.pprint(res)

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


def get_crosstalk_metric(R,T,G,N_PF,N_TF,crosslayer_crosstalk,tf_first_layer,minimize_noncognate_binding,model_folder):
    assert N_PF > 0, "get_crosstalk_metric does not support N_PF = 0"
    # pr_tf_bound is imported with tf_binding_equilibrium
    if tf_first_layer:
        pr_chromatin_open = dill.load(open(os.path.join(model_folder,"tf_chrom_equiv_pr_bound.out"),"rb"))
        pr_chromatin_error = dill.load(open(os.path.join(model_folder,"tf_chrom_equiv_error_rate.out"),"rb"))
    else:
        pr_chromatin_open = dill.load(open(os.path.join(model_folder,"kpr_pr_open.out"), "rb"))
        pr_chromatin_error = dill.load(open(os.path.join(model_folder,"kpr_opening_error_rate.out"),"rb"))

    pr_tf_bound = dill.load(open(os.path.join(model_folder,"tf_pr_bound.out"),"rb"))
    pr_tf_error = dill.load(open(os.path.join(model_folder,"tf_error_rate.out"),"rb"))

    max_concentration = 1e20
    max_expression = pr_chromatin_open(max_concentration,max_concentration)*pr_tf_bound(max_concentration,max_concentration)

    R_bool = (R != 0)
    T_bool = (T != 0)

    def crosstalk_metric(x,c_PF,c_TF,return_var="metric"):
        #if N_PF == 0:   # network is layer 2 TFs only
            #def get_gene_exp(c_PF,c_TF):
                #C_TF = sum(c_TF)
#
                #pr_wrapper = lambda t: pr_tf_bound(C_TF,c_TF[t])
#
                #return list(map(pr_wrapper,T_bool))

        def get_gene_exp(c_PF,c_TF):
            C_PF = sum(c_PF)
            C_TF = sum(c_TF)
            
            if crosslayer_crosstalk:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF+C_TF,c_PF[r])*pr_tf_bound(C_TF+C_PF,c_TF[t])
            else:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF,c_PF[r])*pr_tf_bound(C_TF,c_TF[t])
        
            return np.concatenate(list(map(pr_wrapper,R_bool,T_bool)))/max_expression

        def get_error_frac(c_PF,c_TF):
            C_PF = sum(c_PF)
            C_TF = sum(c_TF)

            if crosslayer_crosstalk:
                E1 = lambda r: pr_chromatin_error(C_PF+C_TF,c_PF[r])
                E2 = lambda t: pr_tf_error(C_TF+C_PF,c_TF[t])
            else:
                E1 = lambda r: pr_chromatin_error(C_PF,c_PF[r])
                E2 = lambda t: pr_tf_error(C_TF,c_TF[t])

            chromatin_error = np.concatenate(list(map(E1,R_bool)))
            tf_error = np.concatenate(list(map(E2,T_bool)))
            total_error = chromatin_error + tf_error - chromatin_error*tf_error
            
            return np.column_stack((chromatin_error,tf_error,total_error))

        if return_var == "gene_exp":
            return get_gene_exp(c_PF,c_TF)
        elif return_var == "error_frac":
            return get_error_frac(c_PF,c_TF)
        elif return_var == "max_expression":
            return max_expression
        elif return_var == "metric":
            if minimize_noncognate_binding:
                gene_exp = get_gene_exp(c_PF,c_TF)
                err_frac = get_error_frac(c_PF,c_TF)[:,2]
                d1 = x - gene_exp*(1-err_frac)
                d2 = gene_exp*err_frac
                return np.transpose(d1)@d1 + np.transpose(d2)@d2
            else:
                d = x - get_gene_exp(c_PF,c_TF)
                return np.transpose(d)@d
        else:
            print(f"unrecognized return variable option {return_var}")
            return

    return crosstalk_metric


def get_crosstalk_metric_from_row(row):
    return get_crosstalk_metric(row["R"],row["T"],row["G"],row["N_PF"],row["N_TF"], \
            row["crosslayer_crosstalk"],row["tf_first_layer"],row["minimize_noncognate_binding"], \
            row["filename"].apply(os.path.dirname))


def get_crosstalk_metric_from_file(filename_in,database,N_PF,N_TF,crosslayer_crosstalk,tf_first_layer,minimize_noncognate_binding,model_folder):
    local_id = extract_local_id(filename_in)

    # load architecture
    R, T, G = get_network(database,local_id)

    return get_crosstalk_metric(R,T,G,N_PF,N_TF,crosslayer_crosstalk,tf_first_layer,minimize_noncognate_binding,model_folder)
