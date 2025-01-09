# Define a function to calculate the square of a number
def calculate_square(n):
    return n ** 2

# Main program
# Ask the user to input a positive integer
num = int(input("Enter a positive integer: "))

# Call the calculate_square function and display the result
square = calculate_square(num)
print(f"The square of {num} is: {square}")
