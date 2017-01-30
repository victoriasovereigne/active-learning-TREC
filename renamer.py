import os
TEXT_DATA_DIR = "/home/nahid/TREC/v4/fr94/12/"
for filename in os.listdir(TEXT_DATA_DIR):
    i = filename.find("z")
    fnewname = filename[0:i]+".z"
    print fnewname
    fpath = os.path.join(TEXT_DATA_DIR, filename)
    fnewpath = os.path.join(TEXT_DATA_DIR, fnewname)
    os.rename(fpath, fnewpath)