# Taking input for marks in three subjects
subject1 = float(input("Enter marks for Subject 1: "))
subject2 = float(input("Enter marks for Subject 2: "))
subject3 = float(input("Enter marks for Subject 3: "))

# Calculating average
average = (subject1 + subject2 + subject3) / 3

# Determining the grade based on the average
if average >= 90:
    grade = "A"
elif 80 <= average < 90:
    grade = "B"
elif 70 <= average < 80:
    grade = "C"
else:
    grade = "Fail"

# Printing the grade
print(f"Average: {average:.2f}")
print(f"Grade: {grade}")
