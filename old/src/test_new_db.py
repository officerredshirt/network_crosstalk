from numpy import *

import shelve
import dill
import sqlite3
import manage_db
import sys, argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "test_new_db",
            description = "",
            epilog = "")
    parser.add_argument("database")

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
    con = sqlite3.connect(database)
    cur = con.cursor()

    # TODO: fix manage_db to print these nicely
    print("PARAMETERS")
    res = manage_db.query_db(database,"SELECT * FROM parameters")
    manage_db.print_res("parameters",res)
    print("")

    print("NETWORKS")
    res = manage_db.query_db(database,"SELECT * FROM networks")
    manage_db.print_res("networks",res)
    print("")

    print("PATTERNS")
    res = manage_db.query_db(database,"SELECT * FROM patterns")
    manage_db.print_res("patterns",res)
    print("")
    
    print("XTALK")
    res = manage_db.query_db(database,"SELECT * FROM xtalk")
    manage_db.print_res("xtalk",res)
    print("")
    
    con.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
