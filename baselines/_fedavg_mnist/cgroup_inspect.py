import subprocess

def get_cgroup_cpu_limit(cgroup_name):
    cpu_shares_path = f"/sys/fs/cgroup/cpu/{cgroup_name}/cpu.shares"
    cpu_quota_path = f"/sys/fs/cgroup/cpu/{cgroup_name}/cpu.cfs_quota_us"
    cpu_period_path = f"/sys/fs/cgroup/cpu/{cgroup_name}/cpu.cfs_period_us"
    
    cpu_shares = subprocess.check_output(["cat", cpu_shares_path]).decode().strip()
    cpu_quota = subprocess.check_output(["cat", cpu_quota_path]).decode().strip()
    cpu_period = subprocess.check_output(["cat", cpu_period_path]).decode().strip()
    
    return {
        "cpu_shares": cpu_shares,
        "cpu_quota": cpu_quota,
        "cpu_period": cpu_period
    }

def get_cgroup_procs(cgroup_name):
    procs_path = f"/sys/fs/cgroup/cpu/{cgroup_name}/cgroup.procs"
    procs = subprocess.check_output(["cat", procs_path]).decode().strip()
    return procs.splitlines()  # Returns a list of PIDs

# Example usage
cgroup_name = "fedops_cid_1"
cpu_limits = get_cgroup_cpu_limit(cgroup_name)
cgroup_procs = get_cgroup_procs(cgroup_name)

print(f"CPU Limits for {cgroup_name}: {cpu_limits}")
print(f"Processes in {cgroup_name}: {cgroup_procs}")
