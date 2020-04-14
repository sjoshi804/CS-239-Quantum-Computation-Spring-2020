# Solution is a function here that checks whether a given function
# f: {0, 1}^n -> {0, 1} is balanced (1 for exactly 1/2 of the inputs
# and 0 for exactly 1/2) or constant (1 or 0 for all inputs)
# Assumption in this code is that f represents its input as a string of
# 0s or 1s
# Input: f - the function, n - the number of bits f takes as input
# Output: 0 if balanced, 1 if constant
def solution(f, n):
    num_of_ones = 0 #Initialize a count for number of ones f returns
    num_of_zeroes = 0 #Initialize a count for number of zeroes f returns
    for i in range(0, 2^(n - 1)): #Loop through 1 more than 50% of inputs
        input = format(i, '0' + str(n) + 'b') #get input in desired format - string of 0s and 1s representing a bit vector
        if f(input): 
            num_of_ones += 1 #increment ones count if f returns 1
        else:
            num_of_zeroes += 1 #increment zeroes count if f returns 0
        if num_of_ones > 0 and num_of_zeroes > 0: #if f has returned both a one and zero at some point, then we know it cannot be constant so it must be balanced, hence return 0
            return 0 
    return 1 #if function has not returned yet then it has returned either 1 or 0 on more than 1/2 the inputs so it must be constant as it can no longer be balanced

def test1(input): #xor on N bits - a balanced function
    N = 10#Parameter to tweak to change how many bits this function is on
    sum = 0
    for i in range(0, N - 1):
        sum += int(input[i])
    return (sum % 2)

def test2(input): #constant function on as many bits as desired
    return 1

assert(solution(test1, 10) == 0)
assert(solution(test2, 10) == 1)
print("Both test cases passed.")