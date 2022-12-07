import os, glob

def get_abs_paths(root_path: str, ext_filter: str = None) -> list:
    try:
        if not(os.path.isdir(root_path)):
            raise ValueError
        if(ext_filter == None):
            list_dirs = glob.glob(os.path.join(root_path, "**", "*"), recursive=True)
            list_dirs.sort()
            if len(list_dirs) == 0:
                return list()
            return list_dirs
        else:
            list_dirs = glob.glob(os.path.join(root_path, "**", "*"+ext_filter), recursive=True)
            list_dirs.sort()
            if len(list_dirs) == 0:
                return list()
            if ext_filter.find('.') == -1:
                raise ValueError
            return list_dirs
    except ValueError as ex:
        print(f"We caught the exception '{ex}'")
            

