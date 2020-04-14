# Solution is a function here that solves the BV problem
# f: {0, 1}^n -> {0, 1} is defined as f(x)= a * x + b
# a - a n bit string, * - inner product mod 2, b - a bit
# Solution must return a and b
# Assumption in code is that the input of f is a string of n 0s & 1s
# Input: f - the function, n - the number of bits f takes as input
# Output: (a, b) # a tuple where a is a string of 0s and 1s and b is an int that is 0 or 1
def solution(f, n):
    b = f(multichar("0", n))
    a = ""
    for i in range(0, n):
        # Generate a string with 1 in the ith place
        input = multichar("0", i) + "1" + multichar("0", n - (i + 1))
        # a*input would return the ith bit of a 0 as the inner product would zero out
        # all but the ith bit - but we xor with b here so to recover a*input we must xor with b again to obtain the a*input
        ith_bit = (f(input) + b) % 2
        a += str(ith_bit)
    return (a, b)
    
# A helper function for generating correctly formatted input
def multichar(text, n):
    return ''.join([x * n for x in text])

def func_f(input):
    sum = 0
    for i in range(0, len(a)):
        sum += int(a[i]) * int(input[i])
    sum += b
    return (sum % 2)

#Adjustable parameters a and b to test with different values
a = "11111" 
b = 1
assert(solution(func_f, len(a)) == (a, b))
print("Test case passed.")