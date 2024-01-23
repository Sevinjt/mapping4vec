reader = open(r"C:\Users\sev_s\Documents\OWL2Vec-Star\logmap_outputlogmap_anchors.txt", "r+")
lines = reader.readlines()

for line in lines:
    dat=line.split("|")

    print("Subject: " , dat[0])
    print ("Object: ", dat[1])
    print ("Predicate depends on confidence:",dat[3] , dat[2] , "and" , dat[4])
    # What to do with confidence!
