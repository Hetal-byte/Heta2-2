# Taking input for the positive integer n
n = int(input("Enter a positive integer: "))

# Initializing the sum variable
sum_of_evens = 0

# Calculating the sum of even numbers
for i in range(2, n + 1, 2):  # Start from 2, step by 2
    sum_of_evens += i

# Printing the result
print(f"The sum of all even numbers between 1 and {n} is: {sum_of_evens}")
