import json
import logging
import os
import subprocess
import time

from tpu_survival import TPUSurvival
from experiments import experiments


project = ''
location = 'us-central1-f'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs/overrunner.log')
fh.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s') # Add timestamp to file logging
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

backup_path = "gs://connors-models/backups"

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("logs/state.json"):
    runners = [TPUSurvival(project=project, location=location, id=i, params=ex) for i, ex in enumerate(experiments)]
else:
    with open("logs/state.json", "r") as f:
        runners = [TPUSurvival(d=state) for state in json.load(f)]

all_runners = runners.copy() # To make sure all TPUs get deleted no matter what

def save(runners):
    states = []
    for r in runners:
        states.append(r.dump_dict())

    with open("logs/state.json", "w") as f:
        json.dump(states, f)


running = time.time()
try:
    while len(runners) > 0:
        for ts in runners:
            if ts.done:
                runners.remove(ts)
                continue

            ts.update_state()
            logging.info("{} - TPU State: {} - Process Running: {}".format(ts.prefix, ts.state, ts.task_running))

            if not ts.created:
                logging.info("{} - Creating TPU".format(ts.prefix))
                ts.create()

            if ts.state == "READY" and not ts.task_running:
                logging.info("{} - Starting Task".format(ts.prefix))
                ts.run_task()

            returncode = None
            if ts.task_running:
                ts.current_process.poll()
                returncode = ts.current_process.returncode

            if returncode is not None:
                logging.info('{} - Training process terminated with code: {}.'.format(ts.prefix,
                    returncode))

                if returncode == 0:
                    # clean up
                    ts.delete()
                    ts.done = True
                    runners.remove(ts)
                    continue

            if ts.state != "READY" and ts.task_running:
                logging.info("{} - Preempted".format(ts.prefix))
                ts.delete()
                ts.kill_current_task()

            if ts.running_time > 60*60*24: # Make a hard checkpoint save every day
                logging.info("Backing up {}".format(ts.prefix))
                subprocess.call(["gsutil", "cp", "-r", ts.params["model_dir"],
                    os.path.join(backup_path, ts.params["model_dir"].split("/")[-1] + "-" + str(ts.current_save))])
                ts.current_save += 1

        if time.time() - running > 60*60: # Save state every hour
            logging.info("Saving state")
            save(runners)
            running = time.time()

        time.sleep(30)

    logging.info("All runners done.")

except Exception as e:
    logging.error("Error occured in check loop: {}".format(str(e)))

finally:
    logging.info("Shutting everything down...")
    for ts in all_runners:
        try:
            ts.delete()
        except Exception as e:
            logging.error(e)
        try:
            ts.kill_current_task()
        except Exception as e:
            logging.error(e)
    save(all_runners)
