
import glob, os
#, output_dir: str, log_file: str, formatter: str = "07d"
def validate_images(input_dir: str):
    try:
        # with open("C:\\Users\\azatv\\VSCProjects\\Second Python\\input") as f:
       list_files = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True) 
       print(list_files)
    except ValueError as ex:
        print("Write an absolute path of the file. The path is not correct. {ex}")
