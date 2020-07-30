import numpy as np

n = 200
h = 0.15
# e = np.repeat(2, n)
e = np.arange(n)
e[29] = 3
e_zero = np.insert(e, 0, 0)
nh = np.floor(n * h)

# def r_code(t):
#     process = np.cumsum(e_zero)
#     process_1 = process[int(nh):]
#     process_2 = process[:(n - int(nh) + 1)]
#     process = process_1 - process_2
#     return process[t]

def r_code(t):
    process = np.cumsum(e)
    print(process)
    process_1 = process[int(nh):]
    process_2 = process[:(n - int(nh) + 1)]
    process = process_1 - process_2
    return process[t]


def paper_code(t):
    # t = t + 1
    proc = np.sum(e[t:t+int(nh)])
    return proc


def my_code():
    proc_a = np.cumsum(e[int(nh)+1:])
    proc_b = np.cumsum(e[:n-int(nh)-1])
    proc = proc_a - proc_b
    return proc

print(r_code(5))
# print(my_code())
print(paper_code(5))

# process = np.cumsum(e_zero)
# process_1 = process[int(nh):]
# print(process_1)

# proc_a = np.cumsum(e[int(nh):])
# print(proc_a)
