from numpy import *

import shelve
import dill
import sqlite3
import manage_db
import sys, argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "check_db_results",
            description = "",
            epilog = "")
    parser.add_argument("database",default="res")

    args = parser.parse_args()
    database = args.database

    ext = os.path.splitext(database)[1]
    if ext == "":
        print("no database file provided--defaulting to local_db.db in " + database)
        database = database + "local_db.db"
    elif not(ext == ".db"):
        print(database + " is not a valid database file")
        sys.exit(1)

    assert os.path.exists(database), "database " + database + " does not exist"

    print("PARAMETERS")
    res = manage_db.query_db(database,"SELECT * FROM parameters")
    print(f"  {len(res)} results")
    print("")

    print("NETWORKS")
    res = manage_db.query_db(database,"SELECT * FROM networks")
    print(f"  {len(res)} results")
    print("")

    print("PATTERNS")
    res = manage_db.query_db(database,"SELECT * FROM patterns")
    print(f"  {len(res)} results")
    print("")
    
    print("XTALK")
    res = manage_db.get_formatted(database,"xtalk")
    err_msg = [x["message"] for x in res if x["success"] == 0]
    print(f"  {len(res)} results, {len(err_msg)} unsuccessful")
    print("unique error messages:")
    print(set(err_msg))
    print("")
    

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
