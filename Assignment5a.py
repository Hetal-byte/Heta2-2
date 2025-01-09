# Ask the user to enter a positive integer
n = int(input("Enter a positive integer: "))

# Using a for loop to print all numbers from 1 to n
print("Numbers from 1 to", n, ":")
for i in range(1, n + 1):
    print(i)

# Using a while loop to calculate the sum of all numbers from 1 to n
sum_of_numbers = 0
counter = 1
while counter <= n:
    sum_of_numbers += counter
    counter += 1

# Print the result
print(f"The sum of all numbers from 1 to {n} is: {sum_of_numbers}")
