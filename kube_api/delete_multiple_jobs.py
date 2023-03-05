from kubernetes import client, config
import time

JOB_NAME = "fedops-client-mjh"
SERVER_JOB_NAME = "fedops-server-mjh"
service_account_name = "fedops-svc-mjh"

JOB_NAME2 = "fedops-client-mjh2"
JOB_NAME3 = "fedops-client-mjh3"

jobs_num = 50


def delete_job(api_instance, job_num: int):
    api_response = api_instance.delete_namespaced_job(
        name=JOB_NAME + str(job_num),
        namespace="fedops",
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    print("Job deleted. status='%s'" % str(api_response.status))


def delete_server_job(api_instance):
    api_response = api_instance.delete_namespaced_job(
        name="fedops-server-mjh",
        namespace="fedops",
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    print("Job deleted. status='%s'" % str(api_response.status))


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_kube_config('config_ssh.txt')
    batch_v1 = client.BatchV1Api()

    # Create a job object with client-python API. The job we
    # created is same as the `pi-job.yaml` in the /examples folder.
    for i in range(jobs_num):
        try:
            delete_job(batch_v1, i)
        except Exception:
            pass
    try:
        delete_server_job(batch_v1)
    except Exception:
        pass

    # delete_job(batch_v1)


if __name__ == '__main__':
    main()