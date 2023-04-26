from numpy import *

import shelve
import sqlite3
import manage_db
import sys, argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "prog name",
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

    x = manage_db.get_formatted(database,"xtalk")
    #print(x[0])
    #x_nan = [y for y in x if y["fun"] == None]
    #print(x_nan)

    err_messages = [y["message"] for y in x]
    print(err_messages)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
