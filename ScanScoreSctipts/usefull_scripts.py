import os
import re
from vars import input_filename, output_filename, directory, input_script, output_script

def filter_txt_file(input_file: str, output_file: str):
    """Filter out all the necessary data to analyze from FileName, ClassID-number, Score, Y values
        Args:
            input_file (str): input file to read from
            output_file (str): output file to write to (automatically create if none)
        """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    for line in lines:
        if "FileName" in line:
            filtered_lines.append(line.strip())
        elif "ClassID-number" in line:
            filtered_lines.append(line.strip())
        elif "Score:" in line:
            filtered_lines.append(line.strip())
        elif "Y values:" in line:
            filtered_lines.append(line.strip())

    with open(output_file, 'w') as file:
        file.write("\n".join(filtered_lines))




def get_highest_score(input_file: str):
    """Method to search for the highest score.
        Args:
            input_file (str): input file to read from
        """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    file_name = ""
    highest_score = None
    for line in lines:
        if "FileName" in line:
            file_name = line.strip()
        elif "Score:" in line:
            score = float(re.search(r"Score:\s+([\d.]+)", line).group(1))
            if highest_score is None or score > highest_score:
                highest_score = score

    return file_name, highest_score

def remove_timestamps(input_file: str, output_file: str):
    """Removing timestamps from txt.
        Args:
            input_file (str): input file to read from
            output_file (str): output file to write to (automatically create if none)
        """
    with open(input_file, 'r') as file:
        content = file.read()

    timestamp_pattern = r"\d+:\d+"
    content_without_timestamps = re.sub(timestamp_pattern, "", content)
    content_without_timestamps = content_without_timestamps.replace("//n","")
    print(content_without_timestamps)
    with open(output_file, 'w+') as file:
        file.write(content_without_timestamps)





def find_positive_ramdas(directory: str):
    """Looks for positive Ramda values and puts them to a list from txt.
        Args:
            directory (str): input file to read from
        """
    positive_ramdas = {}

    for file in os.listdir(directory):
        if file.endswith(".dat"):
            with open(os.path.join(directory, file), "r") as f:
                lines = f.readlines()

            for line in lines:
                if "Ramda" in line:
                    ramda_value = float(re.search(r"Ramda = ([\d.]+)", line).group(1))
                    print(line)
                    if ramda_value > 0:
                        positive_ramdas[file] = ramda_value
                        break

    return positive_ramdas




if __name__ == "__main__":
    filter_txt_file(input_filename, output_filename)
    input_txt_file = "input.txt"
    file_name, highest_score = get_highest_score(output_filename)
    print("Highest Score:", highest_score)
    positive_ramdas = find_positive_ramdas(directory)
    print("Files with Ramda > 0:", positive_ramdas)
    remove_timestamps(input_script, output_script)

