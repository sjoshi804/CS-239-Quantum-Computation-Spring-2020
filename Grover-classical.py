# Solution is a function here that solves the Simon problem
# f: {0, 1}^n -> {0, 1} s.t.
# there exists s in {0,1}^n such that 
# forall x,y: [f(x) = f(y)] iff [(x bitwisexor y) in {0^n, s}].
# i.e. if f(x) = f(y) then x = y or x bitwisexor y = s
# Solution must return s
# Assumption in code is that the input of f is a string of n 0s & 1s
# Input: f - the function, n - the number of bits f takes as input
# Output: s as a number in binary (leading)
import random
def solution(f, n):
    for i in range(2**n): #Loop through all inputs
        x = format(i, '0' + str(n) + 'b') #get input in desired format - string of 0s and 1s representing a bit vector
        
        if (f(x) == 1): #if f evaluates to 1 on x, terminate here
            return 1

    return 0 #if f never = 1 then return 0

def f_has_one(x):
    if x == format(target, '0' + str(N) + 'b'):
        return 1
    else:
        return 0

def f_no_one(x):
    return 0

#Tweakable parameter for how many bits the function operates on
N = 10

#Setting up function
target = random.randint(0, (2**N) - 1) #value for which f evaluates to 1
print(str(N) + " bit function and equal to 1 when x = " + str(target))

#Test Cases
assert(solution(f_has_one, N) == 1)
assert(solution(f_no_one, N) == 0)

print("Test cases passed.")