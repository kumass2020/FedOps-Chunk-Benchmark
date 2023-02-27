from kubernetes import client, config
import time

JOB_NAME = "fedops-client-mjh"
service_account_name = "fedops-svc-mjh"

JOB_NAME2 = "fedops-client-mjh2"
JOB_NAME3 = "fedops-client-mjh3"


def create_containers():
    container_list = []
    cpu = 0
    memory = 0

    for i in range(3):
        container = client.V1Container(
            name=f"fedops-client-{i}",
            image="kumass2020/fedops-client",
            # command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
            resources=client.V1ResourceRequirements(
                requests={"cpu": "2000m", "memory": "1Gi"},
                # limits={"cpu": "8000m", "memory": "8Gi"}
            ),
            volume_mounts=[client.V1VolumeMount(name="airflow-nfs", mount_path="/mnt/fedops-mjh")]
        )
        container_list.append(container)

    return container_list


def create_job_object():
    # Configureate Pod template container
    container_list = create_containers()

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fedops-client-mjh"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container_list[0]],
            volumes=[client.V1Volume(
                name="airflow-nfs",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fedops-pvc-mjh",
                    )
            )]
        ))

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME),
        spec=spec)

    return job


def create_job_object2():
    # Configureate Pod template container
    container_list = create_containers()

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fedops-client-mjh2"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container_list[1]],
            volumes=[client.V1Volume(
                name="airflow-nfs",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fedops-pvc-mjh",
                    )
            )]
        ))

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME2),
        spec=spec)

    return job


def create_job_object3():
    # Configureate Pod template container
    container_list = create_containers()

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fedops-client-mjh3"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container_list[2]],
            volumes=[client.V1Volume(
                name="airflow-nfs",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fedops-pvc-mjh",
                    )
            )]
        ))

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME3),
        spec=spec)

    return job


def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace="fedops")
    print("Job created. status='%s'" % str(api_response.status))
    # get_job_status(api_instance)


def get_job_status(api_instance):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=JOB_NAME,
            namespace="fedops")
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
        time.sleep(1)
        print("Job status='%s'" % str(api_response.status))


def update_job(api_instance, job):
    # Update container image
    job.spec.template.spec.containers[0].image = "perl"
    api_response = api_instance.patch_namespaced_job(
        name=JOB_NAME,
        namespace="default",
        body=job)
    print("Job updated. status='%s'" % str(api_response.status))


def delete_job(api_instance):
    api_response = api_instance.delete_namespaced_job(
        name=JOB_NAME,
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
    job = create_job_object()
    job2 = create_job_object2()
    job3 = create_job_object3()

    create_job(batch_v1, job)
    create_job(batch_v1, job2)
    create_job(batch_v1, job3)

    # update_job(batch_v1, job)

    # delete_job(batch_v1)


if __name__ == '__main__':
    main()