import subprocess
import matplotlib.pyplot as plt
import re
import os


gpu_name_command = ['nvidia-smi', '--query-gpu',
                    'name', '-i', '0', '--format=csv,noheader']
gpu_name_result = subprocess.run(
    gpu_name_command, capture_output=True, text=True)
gpu_name = gpu_name_result.stdout.strip()

print("GPU:", gpu_name)

nvidia_smi_output = subprocess.run(['nvidia-smi', '--query-gpu', 'memory.free',
                                   '-i', '0', '--format=csv,noheader'], capture_output=True, text=True).stdout
free_memory_bytes = 0
if "MiB" in nvidia_smi_output:
    memory_match = re.search(r'(\d+(\.\d+)?) MiB', nvidia_smi_output)
    if memory_match:
        memory = int(memory_match.group(1))
        free_memory_bytes = memory * 1024 * 1024
elif "GiB" in nvidia_smi_output:
    memory_match = re.search(r'(\d+(\.\d+)?) GiB', nvidia_smi_output)
    if memory_match:
        memory = int(float(memory_match.group(1)))
        free_memory_bytes = memory * 1024 * 1024 * 1024
else:
    print("Unknown memory unit in nvidia-smi output")
    exit(1)

print(f"Free memory (gigabytes): {free_memory_bytes / (2**30)}")

num_experiments = 100
min_size = 1
num_steps = 20
step_size = free_memory_bytes / 8 / num_steps
sizes = [int(min_size + i * step_size) for i in range(num_steps)]

experiment_throughputs = []
experiment_sizes = []

for size in sizes:
    command = ['./memcpy', str(size), str(num_experiments)]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip()
    throughput_match = re.search(r'GiB/s (\d+(\.\d+)?)', output)

    if throughput_match:
        throughput = float(throughput_match.group(1))
        experiment_throughputs.append(throughput)
        experiment_sizes.append(size)
    else:
        print(f"Error extracting throughput from output: {output}")
        print(f"Inputs were {' '.join(command)}",)
        break


data_sizes_gbs = [size * 4. / 1024 / 1024 / 1024 for size in experiment_sizes]
times_ms = [data_size / (throughput * 1000) if throughput != 0 else 0 for data_size,
            throughput in zip(data_sizes_gbs, experiment_throughputs)]

gpu_name = gpu_name.replace(" ", "_")
output_dir = f"figures/{gpu_name}"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(data_sizes_gbs, experiment_throughputs, marker='o', linestyle='-')
plt.xlabel('Data Size (GiBs)')
plt.ylabel('Throughput (GiB/s)')
plt.title(f'CUDA Memcpy Throughput vs. Data Size ({gpu_name})')
plt.grid(True)
plt.savefig(f'{output_dir}/memcpy_throughput.png')

plt.figure(figsize=(10, 5))
plt.plot(data_sizes_gbs, times_ms, marker='o', linestyle='-')
plt.xlabel('Data Size (GiBs)')
plt.ylabel('Time (ms)')
plt.title(f'CUDA Memcpy Time vs. Data Size ({gpu_name})')
plt.grid(True)
plt.savefig(f'{output_dir}/memcpy_time.png')
