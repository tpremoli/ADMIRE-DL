from pathlib import Path

def write_batch_to_log(complete_pairs, out_dir, successful_str):
    """Writes a batch of successful scans to a log file.
    
    Args:
        complete_pairs (list): A list of tuples containing the original brain and the new brain files.
        out_dir (str): The output directory.
        successful_str (str): The "success string" to be written to the log file.
    """
    logdir = Path(out_dir, "batches.log").resolve()
    csv_dir = Path(out_dir, "processed.csv").resolve()
    
    with open(logdir, "a") as log:
        log.write(successful_str)
        log.write("\n")
        for ogbrain, newbrain in complete_pairs:
            log.write("\t")
            log.write(Path(ogbrain).resolve().name)
            log.write(" processed to ")
            log.write(Path(newbrain).resolve().name)
            log.write(", ")
            log.write(Path(newbrain).resolve().parent.name)
            log.write("\n")
    
    with open(csv_dir, "a") as csv:
        for ogbrain, newbrain in complete_pairs:
            csv.write('"')
            csv.write(str(Path(ogbrain).resolve()))
            csv.write('"')
            csv.write(",")
            
            csv.write('"')
            csv.write(str(Path(newbrain).resolve()))
            csv.write('"')
            csv.write(",")
            
            csv.write('"')
            csv.write(str(Path(newbrain).resolve().parent.name))
            csv.write('"')

            csv.write("\n")