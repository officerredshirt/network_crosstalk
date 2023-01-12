from numpy import *

import shelve
import dill
import sqlite3
import manage_db
import sys, getopt
import os



def print_usage():
    print("usage is: test_new_db.py -p <folder_in>")

def main(argv):
    folder_in = ""

    try:
        opts, args = getopt.getopt(argv,"hp:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if len(opts) < 1:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-p":
            populate_tables = True
            folder_in = arg

    database = folder_in + "local_db.db"
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
