import os
def chunks(path: str, size: int, **kwargs):
    try:
        if size < 1:
                raise ValueError
        if os.path.isfile(path):
            fl_reader = open(path, kwargs['mode'])
            stick_arr, stick_str = [val for val in fl_reader], ""
            stick_str = "".join(stick_arr)
            quat_c = 0
            for val in range(len(stick_str)):
                if val % 25 == 0 and not val == 0:
                    yield "b'"+(stick_str[quat_c*25:val].replace("\n", "\\r\\n"))+"'"
                    quat_c += 1
                if val % 25 > 0 and val == len(stick_str)-1:
                    yield "b'"+stick_str[quat_c*25:].replace("\n", "\\r\\n")+"'"
        else:
            raise ValueError
    except:
        print("Cought exception")
for i, c in enumerate(chunks("C:\\Users\\azatv\\VSCProjects\\6thUnit\\ex1_data.txt", 25, mode = "r")):
    print(f"Chunk {i} = {c}")




