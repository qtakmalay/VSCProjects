import os, glob

def get_abs_paths(root_path: str, ext_filter: str = None):
    try:
        if(ext_filter == None):
            list_dirs = glob.glob(os.path.join(root_path, "**"), recursive=True)
            list_dirs.sort()
            for val in list_dirs:
                print(val)
        else:
            list_dirs = glob.glob(os.path.join(root_path, "**", "*.py"), recursive=True)
            list_dirs.sort()
            for val in list_dirs:
                print(val)
        if root_path.isdir():
            raise ValueError
    except ValueError as ex:
        print(f"We caught the exception '{ex}'")
            

get_abs_paths("C:\\Users\\azatv\\")
