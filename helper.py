input_file = "./trining.1600000.processed.noemoticon.csv"  
output_file = "helper.csv" 

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if line.endswith(",,,\n"):
            cleaned_line = line[:-4] + "\n" 
        else:
            cleaned_line = line  
        outfile.write(cleaned_line)

print("Last three commas removed successfully where applicable.")