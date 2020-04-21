# Solution is a function here that solves the Simon problem
# f: {0, 1}^n -> {0, 1} s.t.
# there exists s in {0,1}^n such that 
# forall x,y: [f(x) = f(y)] iff [(x bitwisexor y) in {0^n, s}].
# i.e. if f(x) = f(y) then x = y or x bitwisexor y = s
# Solution must return s
# Assumption in code is that the input of f is a string of n 0s & 1s
# Input: f - the function, n - the number of bits f takes as input
# Output: s as a bit vector - string of 0s & 1s

import random 

def solution(f, n):
    hash_table = {}
    for i in range(1, (2**(n - 1)) + 1): #Loop through to test more than half the values for s
        x = format(i, '0' + str(n) + 'b') #get input in desired format - string of 0s and 1s representing a bit vector

        #check if this answer has appeared before
        ans = f(x)
        previous_val = hash_table.get(ans, False)
        if previous_val:
            return bitwise_xor(x, previous_val)
        else:
            hash_table[ans] = x
    return format(i, '0' + str(n) + 'b') #if no answer repeated itself, after more than half the values, then s must be 0

#gets the bitwise xor of two strings of bits
def bitwise_xor(a, b):
    c = ''
    for i in range(0, len(a)):
        c += str((int(a[i])) ^ (int(b[i])))
    return c

def f(x):
    return output[x]

#Adjustable Parameter
n = 10


#setting up function
s = format(random.randint(0, (2**n) - 1), '0' + str(n) + 'b') #random s value
output = {}
for i in range(0, (2**n) - 1):
    x = format(i, '0' + str(n) + 'b')
    if (bitwise_xor(x, s)) in output:
        output[x] = output[bitwise_xor(x, s)]
    else:
        output[x] = x

#Test Case: checks if solution is able to extract s
assert(solution(f, n) == s)

print("Test case passed.")