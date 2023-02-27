from kubernetes import client, config
import time

JOB_NAME = "fedops-server-mjh"
service_account_name = "fedops-svc-mjh"

# def create_containers():
#     container_list = []
#     cpu = 0
#     memory = 0
#
#     for i in range(10):
#         container = client.V1Container(
#             name="fedops-server",
#             image="kumass2020/fedops-server",
#             # command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
#             resources=client.V1ResourceRequirements(
#                 requests={"cpu": "1000m", "memory": "1Gi"},
#                 limits={"cpu": "4000m", "memory": "4Gi"}
#             )
#         )
#         container_list.append(container)


def create_job_object():
    # Configureate Pod template container
    container = client.V1Container(
        name="fedops-server",
        image="kumass2020/fedops-server:10-5-client",
        # command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1000m", "memory": "1Gi"},
            limits={"cpu": "4000m", "memory": "4Gi"}
        )
    )
    # Create and configure a spec section
    # template = client.V1PodTemplateSpec(
    #     metadata=client.V1ObjectMeta(labels={"app": "fedops-mjh"}),
    #     spec=client.V1PodSpec(restart_policy="Never", containers=[container]))

    # Create Service
    service_spec = client.V1ServiceSpec(
        type="LoadBalancer",
        selector={"app": "fedops-mjh"},
        ports=[client.V1ServicePort(port=80, target_port=80)]
    )
    service_meta = client.V1ObjectMeta(name="fedops-service")
    service = client.V1Service(api_version="v1", kind="Service", metadata=service_meta, spec=service_spec)
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fedops-mjh"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            service_account_name=service_account_name
        )
    )

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    # applied with service
    spec.template.metadata.annotations = {"prometheus.io/scrape": "true", "prometheus.io/path": "/metrics"}
    spec.template.metadata.labels = {"app": "fedops-mjh"}
    spec.template.spec.containers[0].ports = [
        client.V1ContainerPort(container_port=8080)
    ]
    spec.template.spec.restart_policy = "Never"
    spec.template.spec.service_account_name = service_account_name
    spec.template.spec.service_account = service_account_name
    spec.template.spec.service = service

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME),
        spec=spec)

    return job


def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace="fedops")
    print("Job created. status='%s'" % str(api_response.status))
    get_job_status(api_instance)


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

    create_job(batch_v1, job)

    # update_job(batch_v1, job)

    # delete_job(batch_v1)


if __name__ == '__main__':
    main()