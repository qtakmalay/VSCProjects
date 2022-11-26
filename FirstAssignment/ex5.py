
line_fill = "#"
width = int(input("Enter width: "))
squat = float(input("Enter squat: "))
b_press = float(input("Enter bench press: "))
deadlift = float(input("Enter deadlift: "))
total = float(squat + b_press + deadlift)
print(f"""{'#':#*{width}}
#{' Powerlifting 2022W': <{width-2}}#
{'#':#^{width}}
Maximum Squat:{squat:>{width-len("Maximum Squat:")-len("kg")}}kg
Maximum Bench Press:{b_press:>{width - len("Maximum Bench Press:") - len("kg")}}kg
Maximum Deadlift:{deadlift:>{width-len("Maximum Deadlift:")-len("kg")}}kg
{'-':-^{width}}
Total:{total:>{width-len("Total:")-len("kg")}}kg""")

