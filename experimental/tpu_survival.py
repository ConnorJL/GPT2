# Adapted from https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/tpu/utils/survival

import json
import logging
from multiprocessing import Process
import os
import shlex
import signal
from subprocess import PIPE
from subprocess import Popen
import time

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


RUN_TASK_COMMAND = 'bash run.sh {tpu_name} {model}'
TENSORFLOW_VERSION = '1.13'
credentials = GoogleCredentials.get_application_default()


class TPUSurvival(object):
    def __init__(self, project=None, location=None, id=None, params=None, d=None):
        if d is None:
            self.project = project
            self.location = location
            self.prefix = params["name"]
            self.id = id
            self.params = params
            self.running_time = 0.
            self.current_save = 0
            self.done = False

        else:
            self.project = d["project"]
            self.location = d["location"]
            self.params = d["params"]
            self.prefix = self.params["name"]
            self.id = d["id"]
            self.running_time = d["running_time"]
            self.current_save = d["current_save"]
            self.done = d["done"]



        # current running job
        self.current_process = None
        self.state = None
        self.created = False
        self.task_running = False


    def tpu_name(self):
        """Format tpu_name to be used in creation and deletion calls."""
        return '{}'.format(self.prefix)

    def tpu_cidr_block(self):
        """Format CIDR block to be used in creation calls."""
        cidr = '10.0.{}.0/29'.format(self.id)
        return cidr

    def update_state(self):
        """Poll the TPU nodes and update self.state."""
        nodes = list_tpus(self.project, self.location).get('nodes', [])
        self.state = None

        for node in nodes:
            name = node['name']
            tpu_name = name.split('/')[-1]
            health = node.get('health', None)
            state = node['state']

            # The node that is running the current task.
            if tpu_name == self.tpu_name():
                # logging.info('{} - TPU health/state: {}: {}/{}'.format(self.prefix, tpu_name, health, state))
                self.state = state

        if self.state is None:
            self.created = False

    def kill_current_task(self):
        """Kill the current running task."""
        logging.info('{} - killing current process: {}'.format(self.prefix, self.current_process.pid))

        # The subprocess runs a shell command, which in turn calls python.
        # This kills the whole process group with the shell command as the
        # process group leader.
        os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)

        self.task_running = False
        self.running_time += time.time() - self.started_time

    # run_task should be called at the beginning and
    # then only after the call to kill current_process
    def run_task(self):
        """Call a subprocess to run the training task on the current TPU node.
        """
        tpu_name = self.tpu_name()

        logging.info('{} - running task: {}'.format(self.prefix, tpu_name))

        with open(self.prefix + ".json", "w") as f:
            json.dump(self.params["model_params"], f)

        cmd = RUN_TASK_COMMAND.format(tpu_name=tpu_name, model=self.prefix + ".json")
        command = shlex.split(cmd)

        # use popen so we can kill it when needed
        p = Popen(command, stdout=PIPE, preexec_fn=os.setsid)

        self.task_running = True
        self.started_time = time.time()

        self.current_process = p

    def delete(self):
        """Delete the TPU node.
        """
        tpu_name = self.tpu_name()

        logging.info('{} - deleting: {}'.format(self.prefix, tpu_name))

        args = (self.project, self.location, tpu_name)
        p = Process(target=delete_tpu, args=args)
        p.start()

        return p

    def create(self):
        """Create a TPU node.
        """
        tpu_name = self.tpu_name()
        tpu_cidr_block = self.tpu_cidr_block()

        logging.info('{} - creating: {}, {}'.format(self.prefix, tpu_name, tpu_cidr_block))

        args = (
            self.project, self.location, tpu_name, self.params["accelerator_type"],
            TENSORFLOW_VERSION, tpu_cidr_block, self.params["preemptible"]
        )
        p = Process(target=create_tpu, args=args)
        p.start()

        self.created = True

        return p

    def dump_dict(self):
        d = {
            "project": self.project,
            "location": self.location,
            "id": self.id,
            "params": self.params,
            "running_time": self.running_time,
            "current_save": self.current_save,
            "done": self.done
        }
        return d

# Util functions

def list_tpus(project, location):
    """List existing TPU nodes in the project/location.
    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    Returns
    A Python dictionary with keys 'nodes' and 'nextPageToken'.
    """
    logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING) # Silence URL spam
    service = discovery.build('tpu', 'v1', credentials=credentials, cache_discovery=False)

    parent = 'projects/{}/locations/{}'.format(project, location)

    request = service.projects().locations().nodes().list(parent=parent)

    return request.execute()


def create_tpu(project, location, tpu_name, accelerator_type='v2-8',
               tensorflow_version='1.11', cidr_block='10.0.101.0',
               preemptible=False):
    """Create a TPU node.
    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.
    accelerator_type: (str) The type of the TPU node, such as "v2-8".
    tensorflow_version: (str) The TensorFlow version, such as "1.11".
    cidr_block: (str) The CIDR block used by the TPU node,
    such as "10.0.101.0".
    preemptible: (bool) Whether the node should be created as preemptible.
    Returns
    A TPU node creation operation object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials, cache_discovery=False)

    parent = 'projects/{}/locations/{}'.format(project, location)

    node = {
        'acceleratorType': accelerator_type,
        'tensorflowVersion': tensorflow_version,
        'network': 'default',
        'cidrBlock': cidr_block,
        'schedulingConfig': {
            'preemptible': preemptible
        }
    }

    # NOTE: in docs and samples nodeId is often referred to as tpu_name
    request = service.projects().locations().nodes().create(
        parent=parent, body=node, nodeId=tpu_name)

    return request.execute()


def get_tpu(project, location, tpu_name):
    """List existing TPU nodes in the project/location.
    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.
    Returns
    A TPU node object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials, cache_discovery=False)

    name = 'projects/{}/locations/{}/nodes/{}'.format(
        project, location, tpu_name)

    request = service.projects().locations().nodes().get(name=name)

    return request.execute()


def delete_tpu(project, location, tpu_name):
    """List existing TPU nodes in the project/location.
    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.
    Returns
    A TPU node deletion operation object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials, cache_discovery=False)

    name = 'projects/{}/locations/{}/nodes/{}'.format(
        project, location, tpu_name)

    request = service.projects().locations().nodes().delete(
        name=name)

    return request.execute()
