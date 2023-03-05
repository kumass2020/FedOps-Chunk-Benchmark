cpu_str = '''
            1 cpu0.15%
            2 cpus9.95%
            3 cpus0.39%
            4 cpus29.51%
            5 cpus0.02%
            6 cpus32.08%
            7 cpus0.02%
            8 cpus19.55%
            9 cpus0.01%
            10 cpus1.91%
            11 cpus0.00%
            12 cpus3.42%
            13 cpus0.00%
            14 cpus1.71%
            15 cpus0.00%
            16 cpus1.06%
            18 cpus0.02%
            20 cpus0.01%
            22 cpus0.00%
            24 cpus0.18%
            26 cpus0.00%
            28 cpus0.00%
            32 cpus0.01%
            36 cpus0.00%
            44 cpus0.00%
            48 cpus0.00%
            56 cpus0.00%
            64 cpus0.00%
            128 cpus0.00%'''
print(cpu_str.replace('\n', '').replace('           ', '').replace('cpu0', 'cpus0'))
cpu_str = cpu_str.replace('\n', '').replace('           ', '').replace('cpu0', 'cpus0')
core_num = []
pie = []
for i in range(29):
    start_idx = cpu_str.index('cpus')
    end_idx = cpu_str.index('%')
    core_num.append(int(cpu_str[1:start_idx-1]))
    pie.append(float(cpu_str[start_idx+4:end_idx]))
    cpu_str = cpu_str[end_idx+1:]

print(core_num)
print(pie)

mean = 0.0
for i in range(29):
    mean += core_num[i] * pie[i]

print(mean/100)