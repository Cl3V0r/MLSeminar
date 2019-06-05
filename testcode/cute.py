file_in = open("../data/postillon.csv", "r")
file_out = open("../data/postillon.txt","w")
for line in file_in:
    if "\"" not in line and "Leserbriefe" not in line and "Postillon" not in line and "Newsticker" not in line and "title" not in line:
        file_out.write(line)
file_in.close()
file_out.close()
